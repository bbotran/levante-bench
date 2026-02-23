"""Model registry: name -> class. Submodules import register from here to avoid circular imports."""

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
