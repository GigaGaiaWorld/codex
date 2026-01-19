from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Type

from pydantic import BaseModel, create_model


def _first_line(doc: str | None) -> str:
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def _build_args_model(func: Callable[..., Any]) -> Type[BaseModel]:
    sig = inspect.signature(func)
    fields: dict[str, tuple[Any, Any]] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[name] = (annotation, default)
    return create_model(f"{func.__name__.title()}Args", **fields)


@dataclass
class SimpleFromDoc:
    general_doc: str
    args_schema: Type[BaseModel]
    signature: inspect.Signature


class FromDoc(SimpleFromDoc):
    @classmethod
    def from_func(cls, func: Callable[..., Any], strict_format: bool = False) -> "FromDoc":
        _ = strict_format
        signature = inspect.signature(func)
        args_schema = _build_args_model(func)
        general_doc = _first_line(func.__doc__)
        return cls(general_doc=general_doc, args_schema=args_schema, signature=signature)
