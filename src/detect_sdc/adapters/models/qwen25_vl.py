"""Qwen2.5-VL model adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .images import load_pil_image


@dataclass
class Qwen25VLAdapter:
    model_path: str
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    answer_suffix: str = "The answer must be limited to 30 words."
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _device: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Qwen25VLAdapter":
        generation = _mapping(config.get("generation"), "generation")
        vision = config.get("vision", {})
        if not isinstance(vision, Mapping):
            raise ValueError("vision must be a mapping")
        return cls(
            model_path=str(config["model_path"]),
            min_pixels=int(vision.get("min_pixels", 256 * 28 * 28)),
            max_pixels=int(vision.get("max_pixels", 1280 * 28 * 28)),
            answer_suffix=str(
                generation.get(
                    "answer_suffix",
                    "The answer must be limited to 30 words.",
                )
            ),
        )

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Qwen25VLAdapter is not loaded")
        return self._model

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        ).eval().to(device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self._device = device

    def generate(
        self,
        question: str,
        image: Any,
        *,
        max_new_tokens: int,
    ) -> str:
        import torch

        if self._model is None or self._processor is None or self._device is None:
            raise RuntimeError("Qwen25VLAdapter must be loaded before generate")
        pil_image = load_pil_image(image)
        messages = [
            {"role": "system", "content": self.answer_suffix},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs.input_ids, output_ids)
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
        )[0].strip()

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
