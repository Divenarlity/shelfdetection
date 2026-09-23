from __future__ import annotations


def validate_model_contract(model, expected_task, required_class: str, label: str):
    """Fail early when a weight file does not implement the production contract."""
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    names = {int(key): str(value) for key, value in names.items()}
    supported = (
        {expected_task} if isinstance(expected_task, str) else set(expected_task)
    )
    if model.task not in supported:
        expected = ", ".join(sorted(repr(task) for task in supported))
        raise RuntimeError(f"{label} task must be one of {expected}; found {model.task!r}")
    if required_class not in names.values():
        raise RuntimeError(
            f"{label} must contain class {required_class!r}; found {list(names.values())}"
        )
    return names
