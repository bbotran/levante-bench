"""VLM adapters for LEVANTE benchmark. Registry: name -> class."""

from levante_bench.models.base import EvalModel, GenEvalModel
from levante_bench.models.registry import (
    get_model_class,
    list_models,
    register,
)

# Import model modules so @register() runs and populates the registry
from levante_bench.models import clip  # noqa: F401


__all__ = [
    "EvalModel",
    "GenEvalModel",
    "get_model_class",
    "list_models",
    "register",
]
