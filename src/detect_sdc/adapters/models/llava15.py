"""LLaVA-v1.5 model adapter."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .images import load_pil_image


@dataclass
class Llava15Adapter:
    model_path: str
    model_base: str | None = None
    source_path: str | None = None
    answer_suffix: str = "The answer must be limited to 30 words."
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _image_processor: Any = field(default=None, init=False, repr=False)
    _model_name: str | None = field(default=None, init=False, repr=False)
    _runtime: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Llava15Adapter":
        generation = _mapping(config.get("generation"), "generation")
        return cls(
            model_path=str(config["model_path"]),
            model_base=(
                None
                if config.get("model_base") in (None, "")
                else str(config["model_base"])
            ),
            source_path=(
                None
                if config.get("source_path") in (None, "")
                else str(config["source_path"])
            ),
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
            raise RuntimeError("Llava15Adapter is not loaded")
        return self._model

    def load(self, device: str) -> None:
        if self.source_path:
            source = Path(self.source_path).expanduser().resolve()
            if not source.is_dir():
                raise FileNotFoundError(f"LLaVA source path does not exist: {source}")
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))

        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_PLACEHOLDER,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import (
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            self.model_path,
            self.model_base,
            model_name,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer
        self._model = model.eval().to(device)
        self._image_processor = image_processor
        self._model_name = model_name
        self._runtime = {
            "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
            "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
            "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
            "IMAGE_PLACEHOLDER": IMAGE_PLACEHOLDER,
            "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
            "conv_templates": conv_templates,
            "process_images": process_images,
            "tokenizer_image_token": tokenizer_image_token,
        }

    def generate(
        self,
        question: str,
        image: Any,
        *,
        max_new_tokens: int,
    ) -> str:
        import torch

        if (
            self._model is None
            or self._tokenizer is None
            or self._image_processor is None
            or self._model_name is None
        ):
            raise RuntimeError("Llava15Adapter must be loaded before generate")

        pil_image = load_pil_image(image)
        prompt = self._build_prompt(
            f"{question.strip()}\n{self.answer_suffix}"
        )
        images_tensor = self._runtime["process_images"](
            [pil_image],
            self._image_processor,
            self._model.config,
        ).to(self._model.device, dtype=torch.float16)
        input_ids = self._runtime["tokenizer_image_token"](
            prompt,
            self._tokenizer,
            self._runtime["IMAGE_TOKEN_INDEX"],
            return_tensors="pt",
        ).unsqueeze(0).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[pil_image.size],
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        return self._tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0].strip()

    def close(self) -> None:
        self._tokenizer = None
        self._model = None
        self._image_processor = None
        self._model_name = None
        self._runtime.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _build_prompt(self, question: str) -> str:
        image_token = (
            self._runtime["DEFAULT_IM_START_TOKEN"]
            + self._runtime["DEFAULT_IMAGE_TOKEN"]
            + self._runtime["DEFAULT_IM_END_TOKEN"]
        )
        if self._runtime["IMAGE_PLACEHOLDER"] in question:
            replacement = (
                image_token
                if self._model.config.mm_use_im_start_end
                else self._runtime["DEFAULT_IMAGE_TOKEN"]
            )
            question = re.sub(
                self._runtime["IMAGE_PLACEHOLDER"],
                replacement,
                question,
            )
        else:
            prefix = (
                image_token
                if self._model.config.mm_use_im_start_end
                else self._runtime["DEFAULT_IMAGE_TOKEN"]
            )
            question = f"{prefix}\n{question}"

        mode = _conversation_mode(self._model_name)
        conversation = self._runtime["conv_templates"][mode].copy()
        conversation.append_message(conversation.roles[0], question)
        conversation.append_message(conversation.roles[1], None)
        return conversation.get_prompt()


def _conversation_mode(model_name: str) -> str:
    lowered = model_name.lower()
    if "llama-2" in lowered:
        return "llava_llama_2"
    if "mistral" in lowered:
        return "mistral_instruct"
    if "v1.6-34b" in lowered:
        return "chatml_direct"
    if "v1" in lowered:
        return "llava_v1"
    if "mpt" in lowered:
        return "mpt"
    return "llava_v0"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
