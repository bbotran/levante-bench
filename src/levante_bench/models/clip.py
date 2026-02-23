"""CLIP-style similarity model adapter."""

from levante_bench.models.registry import register
from levante_bench.models.base import EvalModel


@register("clip_base")
class ClipEvalModel(EvalModel):
    """CLIP ViT-B/32 (or configurable) via transformers."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        **kwargs: object,
    ) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:
            raise ImportError("Install transformers: pip install transformers") from e
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        super().__init__(model=model, processor=processor, device=device)
