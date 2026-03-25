"""Base class for VLM evaluation models."""

from typing import Any, Optional

from PIL import Image


class VLMModel:
    """Base class for all VLM models used in evaluation.

    Subclasses implement load(), generate(), and _build_messages() for
    model-specific behavior. evaluate_trial() and parse_answer() provide
    default implementations that subclasses can override.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load model and processor onto device."""
        ...

    def generate(
        self,
        prompt_text: str,
        images: list[Image.Image] | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate text given a prompt and optional images.

        Args:
            prompt_text: Formatted prompt string from task formatter.
            images: Context and/or option images (order determined by task).
            max_new_tokens: Max tokens to generate.

        Returns:
            Raw generated text from the model.
        """
        ...

    def _build_messages(
        self,
        prompt_text: str,
        images: list[Image.Image] | None = None,
    ) -> list[dict]:
        """Wrap prompt + images into model-specific chat message format.

        Args:
            prompt_text: The prompt string.
            images: Optional images to include in the message.

        Returns:
            List of message dicts in the model's expected chat format.
        """
        ...

    def evaluate_trial(self, trial: dict) -> dict:
        """Run a single trial: generate answer, parse it, return result.

        Args:
            trial: Standard trial dict from VLMDataset with keys:
                prompt, options, option_labels, correct_label,
                context_images, option_images, context_type, option_type

        Returns:
            Dict with: generated_text, predicted_label, correct_label, is_correct
        """
        pass

    def parse_answer(self, generated_text: str, option_labels: list[str]) -> Optional[str]:
        """Extract predicted option label from generated text.

        Default: exact match, starts-with, regex fallback.
        Override in subclass if model has a specific output format.

        Args:
            generated_text: Raw model output.
            option_labels: Valid labels e.g. ["A", "B", "C", "D"].

        Returns:
            Matched label, or None if parsing failed.
        """
        pass
