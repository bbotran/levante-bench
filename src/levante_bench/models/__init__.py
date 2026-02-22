"""VLM adapters for LEVANTE benchmark. Registry: name -> class."""

from levante_bench.models.base import EvalModel, GenEvalModel

_MODEL_REGISTRY: dict[str, type] = {}


def register(name: str):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str) -> type | None:
    return _MODEL_REGISTRY.get(name)


def list_models() -> list[str]:
    return list(_MODEL_REGISTRY.keys())


__all__ = [
    "EvalModel",
    "GenEvalModel",
    "get_model_class",
    "list_models",
    "register",
]
