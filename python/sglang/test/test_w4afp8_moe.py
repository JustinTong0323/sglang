import unittest

import torch

from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod


def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError(
            "the last dim size of int4_values_interleaved tensor must be even."
        )

    input_tensor_int8 = int4_values_interleaved.to(torch.int8)

    low_nibbles = input_tensor_int8[..., 0::2]
    high_nibbles = input_tensor_int8[..., 1::2]

    packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)

    return packed_tensor.to(torch.int8)


def pack_interleave(num_experts, ref_weight, ref_scale, alignment=4):
    n, k = ref_weight.shape[1], ref_weight.shape[2]

    weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8)
    w_q = w_q.contiguous()

    scale_interleaved = ref_scale.reshape(
        ref_scale.shape[0],
        ref_scale.shape[1],
        (ref_scale.shape[2] // alignment),
        alignment,
    )  # [E, N, K/4, 4]
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)  # [E, K/4, N, 4]
    scale_interleaved = scale_interleaved.reshape(
        ref_scale.shape[0],
        ref_scale.shape[2] // alignment,
        ref_scale.shape[1] * alignment,
    )  # [E, K/4, N*4]
    w_scale = scale_interleaved.contiguous()

    return w_q, w_scale


class DummyLayer:
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int, device: str):
        self.num_experts = num_experts
        self.start_expert_id = 0
        self.end_expert_id = num_experts - 1
        # Parameters expected by W4AFp8MoEMethod.apply
        self.w13_weight = torch.empty(
            (num_experts, 2 * intermediate_size, hidden_size // 2), dtype=torch.int8, device=device
        )
        self.w2_weight = torch.empty(
            (num_experts, hidden_size, intermediate_size // 2), dtype=torch.int8, device=device
        )
        self.w13_weight_scale_inv = torch.empty(
            (num_experts, hidden_size // 4, 2 * intermediate_size * 4), dtype=torch.bfloat16, device=device
        )
        self.w2_weight_scale_inv = torch.empty(
            (num_experts, intermediate_size // 4, hidden_size * 4), dtype=torch.bfloat16, device=device
        )
        self.w13_input_scale = torch.ones((1,), dtype=torch.bfloat16, device=device)
        self.w2_input_scale = torch.ones((1,), dtype=torch.bfloat16, device=device)


class TestW4AFp8MoE(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if torch.cuda.get_device_capability()[0] < 9:
            raise unittest.SkipTest("W4AFp8 requires CUDA arch >= 90")
        torch.set_default_device("cuda")

    def _run_case(self, M: int, N: int, K: int, E: int, topk: int, group_size: int = 128):
        dtype = torch.bfloat16
        device = "cuda"

        # Inputs
        a = torch.randn(M, K, dtype=dtype, device=device)
        score = torch.randn((M, E), dtype=dtype, device=device)
        topk_output = select_experts(
            hidden_states=a, router_logits=score, topk_config=TopKConfig(top_k=topk, renormalize=False)
        )
        topk_weights, topk_ids, _ = topk_output

        # Reference weights/scales
        ref_weight_1 = torch.randint(-8, 8, (E, 2 * N, K), dtype=torch.int8, device=device)
        ref_weight_2 = torch.randint(-8, 8, (E, K, N), dtype=torch.int8, device=device)
        affine_coeff = 0.005
        scale_1 = torch.randn(E, 2 * N, K // group_size, dtype=dtype, device=device) * affine_coeff
        scale_2 = torch.randn(E, K, N // group_size, dtype=dtype, device=device) * affine_coeff

        # Pack to kernel layout
        w1_q, w1_scale = pack_interleave(E, ref_weight_1, scale_1)
        w2_q, w2_scale = pack_interleave(E, ref_weight_2, scale_2)

        # Prepare dummy layer and method
        layer = DummyLayer(num_experts=E, hidden_size=K, intermediate_size=N, device=device)
        layer.w13_weight.copy_(w1_q)
        layer.w2_weight.copy_(w2_q)
        layer.w13_weight_scale_inv.copy_(w1_scale.to(torch.bfloat16))
        layer.w2_weight_scale_inv.copy_(w2_scale.to(torch.bfloat16))

        method = W4AFp8MoEMethod(quant_config=None)  # quant_config unused in apply
        # Populate stride buffers similarly to w4afp8.create_weights
        method.a_strides1 = torch.full((E, 3), K, device=device, dtype=torch.int64)
        method.c_strides1 = torch.full((E, 3), 2 * N, device=device, dtype=torch.int64)
        method.a_strides2 = torch.full((E, 3), N, device=device, dtype=torch.int64)
        method.c_strides2 = torch.full((E, 3), K, device=device, dtype=torch.int64)
        method.b_strides1 = method.a_strides1
        method.s_strides13 = method.c_strides1
        method.b_strides2 = method.a_strides2
        method.s_strides2 = method.c_strides2
        method.expert_offsets = torch.empty((E + 1), dtype=torch.int32, device=device)
        method.problem_sizes1 = torch.empty((E, 3), dtype=torch.int32, device=device)
        method.problem_sizes2 = torch.empty((E, 3), dtype=torch.int32, device=device)

        # Run apply
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput

        y = method.apply(layer, StandardDispatchOutput(hidden_states=a, topk_output=topk_output)).hidden_states

        # Reference computation (float8 activations + scales)
        def ref(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for e_idx in range(E):
                mask = topk_ids == e_idx
                activated_tokens = mask.sum(1).bool()
                if activated_tokens.sum() == 0:
                    continue
                x_e = x[activated_tokens, :]
                final_scale = (topk_weights * mask).sum(1)[activated_tokens].unsqueeze(1)

                # Quantize to fp8 with input scale a1 (scalar here)
                a1 = layer.w13_input_scale.to(torch.float32).item()
                x_e = (
                    torch.clamp((x_e / a1), -448.0, 448.0).to(torch.float8_e4m3fn).to(dtype)
                )

                w3_w1 = ref_weight_1[e_idx]
                ref_w_scale_repeat = (
                    scale_1[e_idx].repeat_interleave(128, dim=1).to(float)
                )
                w3_w1 = (w3_w1.to(float) * ref_w_scale_repeat).to(dtype)
                fc1 = (torch.matmul(x_e, w3_w1.T) * layer.w13_input_scale.to(torch.float32)).to(torch.float16)
                gate, fc1 = fc1.chunk(2, dim=-1)
                fc1 = fc1 * torch.nn.functional.silu(gate)

                # Quantize intermediate to fp8 with input scale a2 (scalar here)
                a2 = layer.w2_input_scale.to(torch.float32).item()
                fc1_q = torch.clamp((fc1 / a2), -448.0, 448.0).to(torch.float8_e4m3fn)
                fc1_q = fc1_q.to(dtype)

                w2 = ref_weight_2[e_idx]
                ref_w_scale_repeat = (
                    scale_2[e_idx].repeat_interleave(128, dim=1).to(float)
                )
                w2 = (w2.to(float) * ref_w_scale_repeat).to(dtype)
                fc2 = (torch.matmul(fc1_q, w2.T) * layer.w2_input_scale.to(torch.float32)).to(torch.float16)
                out[activated_tokens, :] += (fc2 * final_scale).to(out.dtype)
            return out

        y_ref = ref(a)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=0.1)

    def test_w4afp8_small(self):
        # Use modest sizes to keep the test fast and compatible with kernel constraints
        self._run_case(M=8, N=512, K=2048, E=8, topk=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

