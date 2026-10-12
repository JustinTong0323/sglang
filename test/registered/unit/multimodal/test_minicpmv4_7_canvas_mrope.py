"""Canvas M-RoPE positions and per-request processing for MiniCPM-V 4.7 (CPU only).

The expected tensors were produced by the checkpoints' reference
``mrope_minicpmv4_7.py`` and pin the port to it.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.minicpmv4_6 import MiniCPMV4_6ImageProcessor
from sglang.srt.multimodal.processors.minicpmv4_7 import (
    MiniCPMV4_7MultimodalProcessor,
    _parse_mm_processor_kwargs,
    _sequential_positions,
    build_image_bounds,
    compute_canvas_positions,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Synthetic ids: the layout code only ever compares them for equality.
IM_START, IM_END, SLICE_START, SLICE_END, NEWLINE, PAD = 900, 901, 902, 903, 904, 905
SPECIAL_IDS = {
    "im_start_id": IM_START,
    "im_end_id": IM_END,
    "slice_start_id": SLICE_START,
    "slice_end_id": SLICE_END,
    "newline_id": NEWLINE,
}


def _canvas(input_ids, target_sizes):
    ids = torch.tensor(input_ids, dtype=torch.long)
    return compute_canvas_positions(
        ids, build_image_bounds(ids, SPECIAL_IDS), target_sizes, SPECIAL_IDS
    )


def _item(modality, start, tgt_size):
    return MultimodalDataItem(
        modality=modality,
        offsets=[(start, start + 1)],
        model_specific_data={"tgt_size": [torch.tensor(tgt_size, dtype=torch.int32)]},
    )


class TestMiniCPMV4_7CanvasMRoPE(CustomTestCase):
    def test_text_only_collapses_to_sequential_positions(self):
        positions = _canvas([10, 11, 12, 13], [])

        self.assertEqual(positions.tolist(), [[0, 1, 2, 3]] * 3)

    def test_thumbnail_grid_places_rows_and_columns(self):
        # 16 visual tokens on a 4x4 grid: <image> halo sits one position below
        # the base, </image> one past the canvas.
        positions = _canvas(
            [10, 11, IM_START] + [PAD] * 16 + [IM_END, 12, 13], [(4, 4)]
        )

        self.assertEqual(positions[0].tolist(), [0, 1, 2] + [2] * 16 + [2, 7, 8])
        self.assertEqual(
            positions[1].tolist(),
            [0, 1, 1] + [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5] + [6, 7, 8],
        )
        self.assertEqual(
            positions[2].tolist(),
            [0, 1, 1] + [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5] + [6, 7, 8],
        )

    def test_slice_is_placed_after_the_thumbnail(self):
        positions = _canvas(
            [10, 11, IM_START]
            + [PAD] * 16
            + [IM_END, SLICE_START]
            + [PAD] * 16
            + [SLICE_END, 12, 13],
            [(4, 4), (4, 4)],
        )

        # A 1x1 slice grid adds no canvas extent, so the slice reuses the
        # thumbnail's coordinates.
        self.assertEqual(positions[0][19], 2)
        self.assertEqual(positions[0][20], 2)
        self.assertEqual(positions[1][21:37].tolist(), positions[1][3:19].tolist())
        self.assertEqual(positions[2][21:37].tolist(), positions[2][3:19].tolist())

    def test_tightly_adjacent_frames_form_one_video_group(self):
        frame = [IM_START] + [PAD] * 16 + [IM_END]
        positions = _canvas([10, 11] + frame * 3 + [12, 13], [(4, 4)] * 3)

        # Each frame advances its own base instead of sharing the first one's.
        self.assertTrue((positions[0][20:36] > positions[0][3:19].max()).all())


class TestMiniCPMV4_7MRoPEPositions(CustomTestCase):
    def _processor(self):
        processor = MiniCPMV4_7MultimodalProcessor.__new__(
            MiniCPMV4_7MultimodalProcessor
        )
        processor.canvas_token_ids = dict(SPECIAL_IDS)
        return processor

    def test_grids_follow_prompt_order_not_item_order(self):
        # Items are appended images-first, but the video frame comes first here.
        input_ids = [10, IM_START] + [PAD] * 4 + [IM_END, 11, IM_START]
        input_ids += [PAD] * 16 + [IM_END, 12]
        items = [_item(Modality.IMAGE, 9, (4, 4)), _item(Modality.VIDEO, 2, (2, 2))]

        positions, delta = self._processor().compute_mrope_positions(input_ids, items)

        expected = _canvas(input_ids, [(2, 2), (4, 4)])
        self.assertEqual(positions.tolist(), expected.tolist())
        self.assertEqual(delta.item(), expected.max().item() + 1 - len(input_ids))

    def test_bounds_and_grid_count_mismatch_falls_back(self):
        input_ids = [10, 11, IM_START] + [PAD] * 16 + [IM_END, 12, 13]
        items = [_item(Modality.IMAGE, 3, (4, 4)), _item(Modality.IMAGE, 3, (4, 4))]

        positions, delta = self._processor().compute_mrope_positions(input_ids, items)

        self.assertEqual(
            positions.tolist(), _sequential_positions(len(input_ids)).tolist()
        )
        self.assertEqual(delta.item(), 0)


class TestCanvasTokenIds(CustomTestCase):
    def test_missing_config_ids_are_filled_from_a_byte_level_tokenizer(self):
        # Byte-level BPE stores "\n" as a different vocab entry, so a literal
        # lookup misses it; ids pinned in the config must still win.
        processor = MiniCPMV4_7MultimodalProcessor.__new__(
            MiniCPMV4_7MultimodalProcessor
        )
        processor.image_start_id = processor.image_end_id = 1
        processor.slice_start_id = processor.slice_end_id = 1
        processor.tokenizer = SimpleNamespace(
            convert_tokens_to_ids=lambda tokens: [None] * len(tokens),
            encode=lambda text, add_special_tokens: [NEWLINE] if text == "\n" else [],
        )
        hf_config = SimpleNamespace(
            mrope_mode="canvas",
            image_start_id=IM_START,
            image_end_id=IM_END,
            slice_start_id=SLICE_START,
            slice_end_id=SLICE_END,
        )

        self.assertEqual(processor._resolve_canvas_token_ids(hf_config), SPECIAL_IDS)


class TestPerRequestProcessorKwargs(CustomTestCase):
    def _processor(self):
        processor = MiniCPMV4_7MultimodalProcessor.__new__(
            MiniCPMV4_7MultimodalProcessor
        )
        processor.image_processor = MiniCPMV4_6ImageProcessor(
            max_slice_nums=9, patch_size=14, downsample_mode="16x"
        )
        return processor

    def test_hf_style_modality_keys_are_flattened(self):
        flat = _parse_mm_processor_kwargs(
            {"images_kwargs": {"max_slice_nums": 1}, "downsample_mode": "4x"}
        )

        self.assertEqual(flat, {"max_slice_nums": 1, "downsample_mode": "4x"})

    def test_non_mapping_kwargs_are_ignored(self):
        self.assertEqual(_parse_mm_processor_kwargs(None), {})
        self.assertEqual(_parse_mm_processor_kwargs("downsample_mode=4x"), {})

    def test_absent_or_invalid_kwargs_keep_the_default(self):
        processor = self._processor()

        for kwargs in (None, {}, {"downsample_mode": "8x", "max_slice_nums": "x"}):
            with self.subTest(kwargs=kwargs):
                self.assertIs(
                    processor._image_processor_for(
                        SimpleNamespace(mm_processor_kwargs=kwargs)
                    ),
                    processor.image_processor,
                )

    def test_kwargs_build_a_request_scoped_image_processor(self):
        processor = self._processor()

        image_processor = processor._image_processor_for(
            SimpleNamespace(
                mm_processor_kwargs={"downsample_mode": "4x", "max_slice_nums": 4}
            )
        )

        self.assertIsNot(image_processor, processor.image_processor)
        self.assertEqual(image_processor.pad_divisor, 4)
        self.assertEqual(image_processor.max_slice_nums, 4)
        # Requests are preprocessed concurrently; the shared default must not move.
        self.assertEqual(processor.image_processor.downsample_mode, "16x")
        self.assertEqual(processor.image_processor.max_slice_nums, 9)

    def test_request_scoped_settings_change_the_cache_identity(self):
        # The pixels are identical under 16x and 4x, but the embeddings are not;
        # sharing a pad value would let the radix cache reuse 16x state for 4x.
        processor = self._processor()
        input_ids = torch.tensor([IM_START, PAD, PAD, IM_END])
        patch = ([torch.ones(3, 14, 28)], [torch.tensor([1, 2], dtype=torch.int32)])
        pad_values = []
        for kwargs in ({}, {"downsample_mode": "4x"}, {"max_slice_nums": 4}):
            image_processor = processor._image_processor_for(
                SimpleNamespace(mm_processor_kwargs=kwargs)
            )
            (item,) = processor._build_items(
                input_ids, PAD, patch, Modality.IMAGE, image_processor
            )
            item.set_pad_value()
            pad_values.append(item.pad_value)

        self.assertEqual(len(set(pad_values)), 3)


if __name__ == "__main__":
    unittest.main()
