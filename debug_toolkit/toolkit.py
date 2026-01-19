from __future__ import annotations

import inspect
import logging
import textwrap
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from typing import Type as ClassType

from pydantic import BaseModel, create_model

from .schema_from_docs import FromDoc, SimpleFromDoc

logger = logging.getLogger(__name__)

_GLOBAL_PARAMS: Dict[str, Any] = {
    "__block_mode__": False,
}
_TOOLKIT_REGISTRY: Dict[str, type["BaseToolKit"]] = {}


def to_dsl_name(py_name: str) -> str:
    """Convert a Pythonic name (snake_case) to DSL name (CamelCase)."""
    parts = py_name.split("_")
    return "".join(word.capitalize() for word in parts)


MISSING = object()


class CtxBinding:
    def __init__(
        self,
        default: Optional[Any] = MISSING,
        default_factory: Optional[Callable[[], Any]] = None,
        required: bool = False,
        allow_none: bool = True,
        alias: Optional[Union[str, Iterable[str]]] = None,
        coerce: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        # required and default cannot both be set.
        if required and (default is not MISSING or default_factory is not None):
            raise ValueError("required=True conflicts with default/default_factory")

        self.default = default
        self.default_factory = default_factory
        self.required = required
        self.allow_none = allow_none

        if isinstance(alias, str):
            self.alias = [alias]
        elif alias is None:
            self.alias = []
        else:
            self.alias = list(alias)

        self.coerce = coerce

    def __set_name__(self, owner: Any, name: str) -> None:
        self._name = name
        self.key = name

    # when getting value of attribute "self.input1"
    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        if obj is None:
            return self
        # if not hasattr(obj, "_binding_dict"), create one, but no binding.
        binding_dict = getattr(obj, "_binding_dict", None)
        if not isinstance(binding_dict, dict):
            binding_dict = {}
            obj._binding_dict = binding_dict

        if not isinstance(binding_dict, dict):
            raise ValueError(f"_binding_dict is not a dict in object {obj}.")

        if self.key in binding_dict:
            value = binding_dict[self.key]
            return self._finalize_value(value, store_default=False, obj=obj)

        for ak in self.alias:
            if ak in binding_dict:
                value = binding_dict[ak]
                return self._finalize_value(value, store_default=False, obj=obj)

        if self.default_factory is not None:
            value = self.default_factory()
            return self._finalize_value(value, store_default=True, obj=obj)

        if self.default is not MISSING:
            return self._finalize_value(self.default, store_default=False, obj=obj)

        if self.required:
            raise ValueError(f"Required binding '{self.key}' not found in binding dict.")

        return None

    # when setting attribute value "self.input1 = value"
    def __set__(self, obj: Any, val: Any) -> None:
        binding_dict = getattr(obj, "_binding_dict", None)
        if not isinstance(binding_dict, dict):
            binding_dict = {}
            obj._binding_dict = binding_dict

        if val is None and not self.allow_none:
            raise ValueError(f"Value for '{self._name}' cannot be None.")

        if self.coerce is not None and val is not None:
            try:
                val = self.coerce(val)
            except Exception as e:
                raise ValueError(f"Failed to coerce value for '{self._name}': {e}")

        binding_dict[self.key] = val

    def _finalize_value(self, val: Any, store_default: bool, obj: Any = None) -> Any:
        if val is None and not self.allow_none:
            raise ValueError(f"Value for '{self._name}' cannot be None.")

        if self.coerce is not None and val is not None:
            try:
                val = self.coerce(val)
            except Exception as e:
                raise ValueError(f"Failed to coerce value for '{self._name}': {e}")

        if store_default and obj is not None:
            try:
                obj._binding_dict[self.key] = val
            except Exception as e:
                logger.warning(f"Failed to store default value for '{self.key}': {e}")

        return val


class BaseToolKit:
    """
    Base class for defining toolkits with multiple tool modes.
    """

    def __init__(self, binding_dict: Dict[str, Any] | None = None) -> None:
        self._binding_dict = binding_dict or {}

    def __init_subclass__(
        cls,
        name: str | None = None,
        description: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init_subclass__(**kw)
        # Step0: register class
        cls._toolkit_name = name or cls.__name__
        _TOOLKIT_REGISTRY[cls._toolkit_name] = cls

        # Step1: set class attributes
        cls._tools_descs: Dict[str, str] = {}
        cls.__doc__ = description or cls.__doc__

        cls._calling_info: Dict[str, List[str]] = {}

    def resolve_tool_calling_input(self) -> Dict[str, Any]:
        """
        Build a JSON schema for a 'tool selector' tool.
        The LLM should return: {"query": ["ToolName1", "ToolName2"]} or {"query": []}.
        """

        tool_desc = """
Description of the toolkit:
{docstring}

Following are the available tools and their descriptions in this toolkit.
Please respond with a list of tool names to select which tool(s) to use, e.g., ["Tool1", "Tool2"].
If none of the tools is suitable, return an empty list [].

Available tools:
{tools}
""".strip()

        tools: Dict[str, str] = {}

        # IMPORTANT: tools live on the class, not instance __dict__
        for _, attr in self.__class__.__dict__.items():
            if hasattr(attr, "_tool_name"):
                tools[attr._tool_name] = attr._description

        tools_text = "\n".join([f"- {k}: {v}" for k, v in tools.items()]) or "- (No tools registered)"

        schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "description": tool_desc.format(
                        docstring=self.__doc__ or "No description provided.",
                        tools=tools_text,
                    ),
                    "items": {"type": "string", "enum": list(tools.keys())},
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        }

        return {"parameters": schema}


def tool_card(
    *,
    expose: Optional[Iterable[str]] = None,
    args_schema: Optional[ClassType[BaseModel]] = None,
    strict_format: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to mark a method as a tool mode with automatic parameter filling from class attributes.
    This decorator is only applicable to methods of BaseToolKit subclasses!!!
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        expose_list = list(expose or [])
        # 1. Inspect function signature:
        sig = inspect.signature(func)
        raw_params = list(sig.parameters.values())
        if not raw_params or raw_params[0].name != "self":
            raise ValueError("This tool can only be used on class methods.")

        agentic_tool_name = to_dsl_name(func.__name__)

        # 2. Validate signature and schema:
        schema_from_doc = FromDoc.from_func(func, strict_format=strict_format)

        general_doc = schema_from_doc.general_doc
        args_model = args_schema or schema_from_doc.args_schema
        call_sig = schema_from_doc.signature

        external_fields: Dict[str, tuple[Any, Any]] = {}
        for name, field_info in args_model.model_fields.items():
            # If name not in expose, we do not include it at all.
            # Case1: in not in expose, just skip:
            if name not in expose_list:
                continue

            # Case3: not in self attributes or not provided by user, filled by llm:
            external_fields[name] = (field_info.annotation, field_info)
        trimmed_args_model = create_model(
            f"{agentic_tool_name}ArgsModel",
            **external_fields,
        )

        @wraps(func)
        def wrapper(self: "BaseToolKit", *args: Any, **kwargs: Any) -> Any:
            # The missing parameters will be filled from self attributes:
            external_bound = call_sig.bind_partial(*args, **kwargs)

            # This decorator is only used for BaseToolKit:
            if not isinstance(self, BaseToolKit):
                raise ValueError(
                    "tool_card decorator can only be used on methods of BaseToolKit subclasses."
                )

            logger.debug(f"Filling parameters for tool method '{func.__name__}':")
            from_self_attrs, from_external, from_agent, from_default = [], [], [], []
            # ======================= Fill in missing parameters ===================== #
            # the clear Priority List for filling parameters:
            # Class attribute value from user > Function Signature default > Class attribute default > raise Error
            for name, param in call_sig.parameters.items():
                if name == "self":
                    continue

                if name in external_bound.arguments:
                    provided = external_bound.arguments[name]
                    if name in expose_list:
                        from_agent.append(name)
                        continue
                    if hasattr(self, name):
                        raise ValueError(
                            f"Parameter '{name}' is provided externally but also exists as class attribute."
                        )
                    from_external.append(name)
                    continue

                # 2) ctx/self attributes:
                if hasattr(self, name):
                    v = getattr(self, name)
                    if v is not None:
                        external_bound.arguments[name] = v
                        from_self_attrs.append(name)
                        continue

                # 3) Function signature default:
                if param.default is not inspect.Parameter.empty:
                    external_bound.arguments[name] = param.default
                    from_default.append(name)
                    continue

                raise ValueError(f"Missing required param '{name}'")

            calling_info = f"""For tool method '{func.__name__}':
            - Exposed to agent: {from_agent} => {[external_bound.arguments[n] for n in from_agent]}
            - Filled from self attributes: {from_self_attrs} => {[getattr(self, n) for n in from_self_attrs]}
            - Filled from externals: {from_external} => {[external_bound.arguments[n] for n in from_external]}
            - Filled from defaults: {from_default} => {[external_bound.arguments[n] for n in from_default]}
            """
            self._calling_info[func.__name__] = textwrap.dedent(calling_info).strip()
            logger.debug(calling_info)

            # ======================= Process outputs ======================== #
            results = func(self, **external_bound.arguments)
            return results

        wrapper._tool_name = agentic_tool_name
        wrapper._description = general_doc
        wrapper._args_model = trimmed_args_model

        return wrapper

    return decorator
