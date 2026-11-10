"""Resolve the reference for an operator on a Torch device."""

from importlib import import_module

from .add import reference as add_reference

_DEFAULT_REFERENCES = {"add": add_reference}
_BACKEND_REFERENCES = {}


def get_reference(op_name, device):
    device_type = str(device).split(":", 1)[0]
    reference = _BACKEND_REFERENCES.get((op_name, device_type))
    if reference is None:
        reference = _DEFAULT_REFERENCES.get(op_name)
    if reference is None:
        module_name = f"{__name__}.{op_name}"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
        else:
            reference = getattr(module, "reference", None)
    if not callable(reference):
        raise LookupError(f"No predefined reference for {op_name!r} on {device_type!r}")
    return reference
