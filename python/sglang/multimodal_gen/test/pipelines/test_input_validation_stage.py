import unittest

from PIL import Image

from sglang.multimodal_gen.configs.sample.base import DataType
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import Req
from sglang.multimodal_gen.runtime.pipelines.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class TestInputValidationStage(unittest.TestCase):
    def test_qwen_edit_preserves_requested_dimensions(self):
        stage = InputValidationStage()
        server_args = ServerArgs.from_kwargs(
            model_path="Qwen/Qwen-Image-Edit",
            num_gpus=1,
        )

        batch = Req(
            data_type=DataType.IMAGE,
            prompt="test prompt",
            seed=1,
            width=1024,
            height=1536,
            num_outputs_per_prompt=1,
        )
        batch.pil_image = Image.new("RGB", (1024, 1536))
        batch.width_not_provided = False
        batch.height_not_provided = False

        updated = stage.forward(batch, server_args)

        self.assertEqual(updated.width, 1024)
        self.assertEqual(updated.height, 1536)


if __name__ == "__main__":
    unittest.main()
