import io
import unittest
from pathlib import Path

from PIL import Image

from detect_sdc.adapters import load_model_adapter
from detect_sdc.adapters.models.images import load_pil_image
from detect_sdc.adapters.models.llava15 import Llava15Adapter
from detect_sdc.adapters.models.qwen25_vl import Qwen25VLAdapter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ModelAdapterTest(unittest.TestCase):
    def test_qwen_adapter_loads_from_config_without_loading_weights(self):
        adapter = load_model_adapter(
            REPOSITORY_ROOT / "configs/models/qwen25_vl.yaml"
        )

        self.assertIsInstance(adapter, Qwen25VLAdapter)
        self.assertEqual(adapter.min_pixels, 256 * 28 * 28)
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            _ = adapter.model

    def test_llava_adapter_loads_from_config_without_llava_import(self):
        adapter = load_model_adapter(
            REPOSITORY_ROOT / "configs/models/llava15.yaml"
        )

        self.assertIsInstance(adapter, Llava15Adapter)
        self.assertTrue(Path(adapter.source_path).is_dir())
        self.assertIn("30 words", adapter.answer_suffix)
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            _ = adapter.model

    def test_image_normalization_supports_embedded_parquet_bytes(self):
        image = Image.new("RGB", (2, 2), color="red")
        stream = io.BytesIO()
        image.save(stream, format="PNG")

        loaded = load_pil_image({"bytes": stream.getvalue(), "path": None})

        self.assertEqual(loaded.mode, "RGB")
        self.assertEqual(loaded.size, (2, 2))


if __name__ == "__main__":
    unittest.main()
