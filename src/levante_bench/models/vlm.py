"""VLM model implementations. One class per model family."""

from typing import Optional

from PIL import Image

from levante_bench.models.base import VLMModel
from levante_bench.models.registry import register


@register("smolvlm2")
class SmolVLM2Model(VLMModel):
    """SmolVLM2 via HuggingFace AutoProcessor + AutoModelForVision2Seq."""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", device: str = "cpu") -> None:
        super().__init__(model_name=model_name, device=device)

    def load(self) -> None:
        """Load SmolVLM2 model and processor from HuggingFace."""
        pass

    def generate(
        self,
        prompt_text: str,
        images: list[Image.Image] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate text using SmolVLM2. Applies chat template then decodes.

        Args:
            prompt_text: Formatted prompt from task formatter.
            images: Optional images (context and/or options).
            max_new_tokens: Max tokens to generate.

        Returns:
            Decoded generated text (new tokens only).
        """
        pass

    def _build_messages(
        self,
        prompt_text: str,
        images: list[Image.Image] | None = None,
    ) -> list[dict]:
        """Build SmolVLM2 chat messages with image tokens.

        Args:
            prompt_text: The prompt string.
            images: Optional PIL images to embed in the message.

        Returns:
            Messages in SmolVLM2's expected format:
            [{"role": "user", "content": [{"type": "image", ...}, {"type": "text", ...}]}]
        """
        pass
