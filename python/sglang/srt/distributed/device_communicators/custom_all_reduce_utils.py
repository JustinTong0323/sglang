# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/custom_all_reduce_utils.py

import ctypes
import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
from functools import wraps
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence, TypeVar

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing_extensions import ParamSpec

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()

if _is_cuda:
    try:
        import pynvml
    except ImportError as e:
        logger.warning("Failed to import pynvml with %r", e)

if _is_hip:
    try:
        from amdsmi import (
            AmdSmiException,
            amdsmi_get_processor_handles,
            amdsmi_init,
            amdsmi_shut_down,
            amdsmi_topo_get_link_type,
        )
    except ImportError as e:
        logger.warning("Failed to import amdsmi with %r", e)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def update_environment_variables(envs: Dict[str, str]):
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s " "from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v


def producer(
    batch_src: Sequence[int],
    producer_queue,
    consumer_queue,
    result_queue,
    cuda_visible_devices: Optional[str] = None,
):
    if cuda_visible_devices is not None:
        update_environment_variables({"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    lib = CudaRTLibrary()
    for i in batch_src:
        lib.cudaSetDevice(i)
        pointer = lib.cudaMalloc(1024)
        lib.cudaMemset(pointer, 1, 1024)
        lib.cudaDeviceSynchronize()
        handle = lib.cudaIpcGetMemHandle(pointer)
        producer_queue.put(handle)
        open_success = consumer_queue.get()
        if open_success:
            # use two queues to simulate barrier
            producer_queue.put(0)
            consumer_queue.get()
            # check if the memory is modified
            host_data = (ctypes.c_char * 1024)()
            lib.cudaMemcpy(host_data, pointer, 1024)  # type: ignore
            for i in range(1024):
                if ord(host_data[i]) != 2:
                    open_success = False
                    break
        result_queue.put(open_success)
        lib.cudaDeviceReset()


def consumer(
    batch_tgt: Sequence[int],
    producer_queue,
    consumer_queue,
    result_queue,
    cuda_visible_devices: Optional[str] = None,
):
    if cuda_visible_devices is not None:
        update_environment_variables({"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    lib = CudaRTLibrary()
    for j in batch_tgt:
        lib.cudaSetDevice(j)
        handle = producer_queue.get()
        open_success = False
        try:
            pointer = lib.cudaIpcOpenMemHandle(handle)  # type: ignore
            open_success = True
        except RuntimeError:
            # cannot error out here, because the producer process
            # is still waiting for the response.
            pass
        consumer_queue.put(open_success)
        if open_success:
            # modify the memory
            lib.cudaMemset(pointer, 2, 1024)
            lib.cudaDeviceSynchronize()
            # use two queues to simulate barrier
            producer_queue.get()
            consumer_queue.put(0)
            # check if the memory is modified
            host_data = (ctypes.c_char * 1024)()
            lib.cudaMemcpy(host_data, pointer, 1024)  # type: ignore
            for i in range(1024):
                if ord(host_data[i]) != 2:
                    open_success = False
                    break
        result_queue.put(open_success)
        lib.cudaDeviceReset()


def can_actually_p2p(
    batch_src: Sequence[int],
    batch_tgt: Sequence[int],
) -> Sequence[bool]:
    """
    Usually, checking if P2P access is enabled can be done by
    `torch.cuda.can_device_access_peer(src, tgt)`. However, sometimes
    the driver might be broken, and `torch.cuda.can_device_access_peer(src, tgt)`
    returns `True` even if P2P access is not actually possible.
    See https://github.com/vllm-project/vllm/issues/2728 and
    https://forums.developer.nvidia.com/t/direct-gpu-gpu-communication-does-not-seem-to-work-properly/283264/10
    Therefore, we have to perform a real P2P access to check if it is actually
    possible.

    Note on p2p and cuda IPC:
    Usually, one process uses one GPU:
    GPU src --> cuda context src --> tensor src --> process src

    We need to combine p2p and cuda IPC, so that:
    GPU src --> cuda context src --> tensor src --> process src
                                      |shared|
    GPU tgt --> cuda context tgt --> tensor tgt --> process tgt
    That is to say, process src creates a tensor in GPU src, passes IPC handle to
    process tgt, and process tgt accesses the tensor in GPU tgt. Any operation on the
    tensor in process tgt will be reflected in the tensor in process src, because
    they are the same memory segment.
    It is important to note that process tgt accesses the tensor in GPU tgt, not
    GPU src. That's why we need p2p access.

    The most time-consuming part is the process creation. To avoid creating
    processes for every pair of GPUs, we use batched testing. We create two
    processes for testing all pairs of GPUs in batch. The trick is to reset
    the device after each test (which is not available in PyTorch).
    """  # noqa
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    # pass the CUDA_VISIBLE_DEVICES to the child process
    # to make sure they see the same set of GPUs

    # make sure the processes are spawned
    smp = mp.get_context("spawn")
    producer_queue = smp.Queue()
    consumer_queue = smp.Queue()
    result_queue = smp.Queue()
    p_src = smp.Process(
        target=producer,
        args=(
            batch_src,
            producer_queue,
            consumer_queue,
            result_queue,
            cuda_visible_devices,
        ),
    )
    p_tgt = smp.Process(
        target=consumer,
        args=(
            batch_tgt,
            producer_queue,
            consumer_queue,
            result_queue,
            cuda_visible_devices,
        ),
    )
    p_src.start()
    p_tgt.start()
    p_src.join()
    p_tgt.join()
    assert p_src.exitcode == 0 and p_tgt.exitcode == 0
    result: List[bool] = []
    for src, tgt in zip(batch_src, batch_tgt):
        a = result_queue.get()
        b = result_queue.get()
        if a != b:
            logger.warning(
                "Two processes do not agree on the P2P access"
                " status on %d -> %d, treat as disabled.",
                src,
                tgt,
            )
            result.append(False)
        else:
            result.append(a)
    return result


# why do we need this cache?
# we are testing peer-to-peer (p2p) access between GPUs,across processes.
# if we test it every time, it will be very slow, because we need to create
#  N * N * 2 processes, where N is the world size. This is very slow.
# to reduce the time, we use a cache file to store the p2p access status.
# the cache file is generated by the master process if it does not exist.
# then all the processes can read the cache file to check the p2p access status.
# Note that the cache file is suffixed by the CUDA_VISIBLE_DEVICES, so that we
#  can have different cache files for different CUDA_VISIBLE_DEVICES settings,
#  e.g. used by different vllm engines. The device id in the cache file is a
#  **local** device id, i.e. from 0 to num_dev-1, where num_dev is the number
#  of visible devices in the vllm engine.
_gpu_p2p_access_cache: Optional[Dict[str, bool]] = None


def gpu_p2p_access_check(src: int, tgt: int) -> bool:
    """Check if GPU src can access GPU tgt."""

    # if the cache variable is already calculated,
    # read from the cache instead of checking it again
    global _gpu_p2p_access_cache
    if _gpu_p2p_access_cache is not None:
        return _gpu_p2p_access_cache[f"{src}->{tgt}"]

    is_distributed = dist.is_initialized()

    num_dev = torch.cuda.device_count()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices is None:
        cuda_visible_devices = ",".join(str(i) for i in range(num_dev))

    # VLLM_CACHE_ROOT -> SGLANG_CACHE_ROOT
    # "~/.cache/vllm" -> "~/.cache/sglang"
    SGLANG_CACHE_ROOT = os.path.expanduser("~/.cache/sglang")
    path = os.path.join(
        SGLANG_CACHE_ROOT, f"gpu_p2p_access_cache_for_{cuda_visible_devices}.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from sglang.srt.distributed.parallel_state import get_world_group

    if (not is_distributed or get_world_group().local_rank == 0) and (
        not os.path.exists(path)
    ):
        # only the local master process (with local_rank == 0) can
        #  enter this block to calculate the cache
        logger.info("generating GPU P2P access cache in %s", path)
        cache: Dict[str, bool] = {}
        ids = list(range(num_dev))
        # batch of all pairs of GPUs
        batch_src, batch_tgt = zip(*list(product(ids, ids)))
        # NOTE: we use `subprocess` rather than `multiprocessing` here
        # because the caller might not have `if __name__ == "__main__":`,
        # in that case we cannot use spawn method in multiprocessing.
        # However, `can_actually_p2p` requires spawn method.
        # The fix is, we use `subprocess` to call the function,
        # where we have `if __name__ == "__main__":` in this file.

        # use a temporary file to store the result
        # we don't use the output of the subprocess directly,
        # because the subprocess might produce logging output
        with tempfile.NamedTemporaryFile() as output_file:
            input_bytes = pickle.dumps((batch_src, batch_tgt, output_file.name))
            returned = subprocess.run(
                [sys.executable, __file__], input=input_bytes, capture_output=True
            )
            # check if the subprocess is successful
            try:
                returned.check_returncode()
            except Exception as e:
                # wrap raised exception to provide more information
                raise RuntimeError(
                    f"Error happened when batch testing "
                    f"peer-to-peer access from {batch_src} to {batch_tgt}:\n"
                    f"{returned.stderr.decode()}"
                ) from e
            with open(output_file.name, "rb") as f:
                result = pickle.load(f)
        for _i, _j, r in zip(batch_src, batch_tgt, result):
            cache[f"{_i}->{_j}"] = r
        with open(path, "w") as f:
            json.dump(cache, f, indent=4)
    if is_distributed:
        get_world_group().barrier()
    logger.info("reading GPU P2P access cache from %s", path)
    with open(path) as f:
        cache = json.load(f)
    _gpu_p2p_access_cache = cache
    return _gpu_p2p_access_cache[f"{src}->{tgt}"]


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if _is_hip:
            try:
                amdsmi_init()
                return fn(*args, **kwargs)
            finally:
                amdsmi_shut_down()
        else:
            pynvml.nvmlInit()
            try:
                return fn(*args, **kwargs)
            finally:
                pynvml.nvmlShutdown()

    return wrapper


@with_nvml_context
def is_full_nvlink(physical_device_ids: List[int], world_size: int) -> bool:
    if _is_hip:
        """
        query if the set of gpus are fully connected by xgmi (1 hop)
        """
        handles = [amdsmi_get_processor_handles()[i] for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        link_type = amdsmi_topo_get_link_type(handle, peer_handle)
                        # type is 2 for XGMI
                        if link_type["hops"] != 1 or link_type["type"] != 2:
                            return False
                    except AmdSmiException as error:
                        logger.error("AMD 1 hop XGMI detection failed.", exc_info=error)
                        return False
        return True
    else:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                        )
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        logger.exception(
                            "NVLink detection failed. This is normal if your"
                            " machine has no NVLink equipped."
                        )
                        return False
        return True


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


__all__ = ["gpu_p2p_access_check"]

if __name__ == "__main__":
    batch_src, batch_tgt, output_file = pickle.loads(sys.stdin.buffer.read())
    result = can_actually_p2p(batch_src, batch_tgt)
    with open(output_file, "wb") as f:
        f.write(pickle.dumps(result))
