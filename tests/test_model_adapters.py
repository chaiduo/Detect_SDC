import io
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from detect_sdc.adapters import load_model_adapter
from detect_sdc.adapters.models.images import load_pil_image
from detect_sdc.adapters.models.internvl3 import InternVL3Adapter
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

    def test_internvl3_adapter_loads_from_config_without_loading_weights(self):
        adapter = load_model_adapter(
            REPOSITORY_ROOT / "configs/models/internvl3.yaml"
        )

        self.assertIsInstance(adapter, InternVL3Adapter)
        self.assertEqual(adapter.image_size, 448)
        self.assertEqual(adapter.max_tiles, 12)
        self.assertIn("30 words", adapter.answer_suffix)
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            _ = adapter.model

    def test_internvl3_generation_sets_pad_token_id(self):
        class FakePixels:
            def to(self, **_kwargs):
                return self

        class FakeChatModel:
            device = "cuda:0"

            def __init__(self):
                self.generation_config = None

            def chat(
                self,
                _tokenizer,
                _pixel_values,
                _prompt,
                generation_config,
            ):
                self.generation_config = generation_config
                return "answer"

        adapter = InternVL3Adapter(model_path="/unused")
        adapter._chat_model = FakeChatModel()
        adapter._tokenizer = SimpleNamespace(
            pad_token_id=151645,
            eos_token_id=151645,
        )
        adapter._load_pixel_values = lambda image: FakePixels()

        answer = adapter.generate("question", object(), max_new_tokens=50)

        self.assertEqual(answer, "answer")
        self.assertEqual(
            adapter._chat_model.generation_config["pad_token_id"],
            151645,
        )

    def test_image_normalization_supports_embedded_parquet_bytes(self):
        image = Image.new("RGB", (2, 2), color="red")
        stream = io.BytesIO()
        image.save(stream, format="PNG")

        loaded = load_pil_image({"bytes": stream.getvalue(), "path": None})

        self.assertEqual(loaded.mode, "RGB")
        self.assertEqual(loaded.size, (2, 2))


if __name__ == "__main__":
    unittest.main()
