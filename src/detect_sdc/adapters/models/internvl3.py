"""InternVL3 model adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .images import load_pil_image


@dataclass
class InternVL3Adapter:
    model_path: str
    image_size: int = 448
    min_tiles: int = 1
    max_tiles: int = 12
    use_thumbnail: bool = True
    torch_dtype: str = "bfloat16"
    low_cpu_mem_usage: bool = True
    use_flash_attn: bool = True
    load_in_8bit: bool = False
    device_map: Any = None
    answer_suffix: str = "The answer must be limited to 30 words."
    _chat_model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _device: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "InternVL3Adapter":
        generation = _mapping(config.get("generation"), "generation")
        vision = config.get("vision", {})
        runtime = config.get("runtime", {})
        if not isinstance(vision, Mapping):
            raise ValueError("vision must be a mapping")
        if not isinstance(runtime, Mapping):
            raise ValueError("runtime must be a mapping")
        return cls(
            model_path=str(config["model_path"]),
            image_size=int(vision.get("image_size", 448)),
            min_tiles=int(vision.get("min_tiles", 1)),
            max_tiles=int(vision.get("max_tiles", 12)),
            use_thumbnail=bool(vision.get("use_thumbnail", True)),
            torch_dtype=str(runtime.get("torch_dtype", "bfloat16")),
            low_cpu_mem_usage=bool(runtime.get("low_cpu_mem_usage", True)),
            use_flash_attn=bool(runtime.get("use_flash_attn", True)),
            load_in_8bit=bool(runtime.get("load_in_8bit", False)),
            device_map=runtime.get("device_map"),
            answer_suffix=str(
                generation.get(
                    "answer_suffix",
                    "The answer must be limited to 30 words.",
                )
            ),
        )

    @property
    def model(self) -> Any:
        if self._chat_model is None:
            raise RuntimeError("InternVL3Adapter is not loaded")
        return getattr(self._chat_model, "language_model", self._chat_model)

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = _torch_dtype(torch, self.torch_dtype)
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "use_flash_attn": self.use_flash_attn,
            "trust_remote_code": True,
        }
        if self.load_in_8bit:
            kwargs["load_in_8bit"] = True
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map

        model = AutoModel.from_pretrained(self.model_path, **kwargs).eval()
        if self.device_map is None and not self.load_in_8bit:
            model = model.to(device)

        self._chat_model = model
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
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

        if self._chat_model is None or self._tokenizer is None:
            raise RuntimeError("InternVL3Adapter must be loaded before generate")

        pixel_values = self._load_pixel_values(image).to(
            device=self._chat_model.device,
            dtype=_torch_dtype(torch, self.torch_dtype),
        )
        prompt = f"{question.strip()}\n{self.answer_suffix}"
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        }
        return str(
            self._chat_model.chat(
                self._tokenizer,
                pixel_values,
                prompt,
                generation_config,
            )
        ).strip()

    def close(self) -> None:
        self._chat_model = None
        self._tokenizer = None
        self._device = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load_pixel_values(self, image: Any) -> Any:
        import torch

        pil_image = load_pil_image(image)
        transform = _build_transform(self.image_size)
        tiles = _dynamic_preprocess(
            pil_image,
            min_num=self.min_tiles,
            max_num=self.max_tiles,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )
        return torch.stack([transform(tile) for tile in tiles])


def _build_transform(input_size: int) -> Any:
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _dynamic_preprocess(
    image: Any,
    *,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
) -> list[Any]:
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        original_width,
        original_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized = image.resize((target_width, target_height))
    processed = []
    tiles_per_row = target_width // image_size
    for index in range(blocks):
        left = (index % tiles_per_row) * image_size
        top = (index // tiles_per_row) * image_size
        processed.append(
            resized.crop(
                (
                    left,
                    top,
                    left + image_size,
                    top + image_size,
                )
            )
        )
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio = (1, 1)
    best_ratio_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _torch_dtype(torch_module: Any, name: str) -> Any:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
