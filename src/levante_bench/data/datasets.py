"""PyTorch/NumPy datasets for evaluation."""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class LevanteDataset(Dataset):
    """
    Dataset from a manifest DataFrame with columns image1..imageN, text1.
    Paths can be absolute or relative to base_path.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        base_path: Path | str | None = None,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.base_path = Path(base_path) if base_path else None
        self.num_image_cols = len([c for c in self.manifest.columns if _is_image_col(c)])
        self.num_text_cols = len([c for c in self.manifest.columns if _is_text_col(c)])

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        images = []
        for i in range(1, self.num_image_cols + 1):
            col = f"image{i}"
            if col not in row:
                break
            path = row[col]
            if pd.isna(path) or not path:
                break
            path = Path(path)
            if self.base_path and not path.is_absolute():
                path = self.base_path / path
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception:
                images.append(Image.new("RGB", (224, 224), color=(128, 128, 128)))
        texts = []
        for i in range(1, self.num_text_cols + 1):
            col = f"text{i}"
            if col not in row:
                break
            texts.append(str(row[col]) if not pd.isna(row[col]) else "")
        return {"images": images, "text": texts}


def _is_image_col(c: str) -> bool:
    import re
    return bool(re.match(r"^image[0-9]+$", c))


def _is_text_col(c: str) -> bool:
    import re
    return bool(re.match(r"^text[0-9]+$", c))


def collate_levante(batch: list[dict]) -> dict:
    """Collate batch: list of dicts -> dict of lists (flatten images; one text per trial)."""
    return {
        "images": [img for ex in batch for img in ex["images"]],
        "text": [ex["text"][0] if ex["text"] else "" for ex in batch],
    }
