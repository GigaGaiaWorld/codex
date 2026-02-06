from __future__ import annotations
from enum import Enum
from typing import Any, List, Dict, Optional, Union
from pydantic import BaseModel, Field
from ..models.models import (
    HashID,
    NameType,
    VarType,
    QuotedStr,
    VarAnoType,
    BuildVarType,
    RTVarType,
    BuildVarTypeAdapter,
    RTVarTypeAdapter,
    is_valid,
)

import warnings
"""
This module defines core data structures for predicates, including:
    - Initials for parsed structural info
    - Builtins for reserved builtin slots
    - Arguments for unified argument representation
They will be composed into IRTParameterDict (Internal Representation of Intermediate Parameters Dict),
which serves as the canonical internal representation of predicate parameters.

This Intermediate Representation (IR) is expected to be used '''across all stages''', 
maybe we will put it elsewhere in the future when we build the agent parts.

Some notes on the design:
1) We separate 'build' and 'runtime' arguments explicitly in IRTParameterDict,
"""

# ============================= Stage Enum ============================= #
# class Stage(str, Enum):
#     INIT = "init"        # layer0: parsing/meta
#     BUILD = "build"      # layer1: graph construction
#     RUNTIME = "runtime"  # layer2: runtime execution


# ============================= Type Hint ============================== #
class ArgTypeHint(BaseModel):
    """
    Lightweight type hint for arguments.
    This is descriptive only and does not enforce runtime casting here.
    """
    name: str = Field(..., examples=["var", "link", "path", "ref", "hint", "schema"])
    schemas: Optional[Dict[str, Any]] = None  # e.g. structured JSON schema, API return fields, etc.

# =============================== Initials ============================= #
class Initials(BaseModel):
    """
    Parsed structural information from the predicate:
    - stable meta about head and agent context
    """
    # stage: Stage = Field(default=Stage.INIT)

    hash_id: Optional[HashID] = None

    head_name: Optional[NameType] = None
    head_args: List[VarType | NameType | QuotedStr] = Field(default_factory=list)
    head_ctx: Optional[str] = None

    # Agent source info (e.g. langda(...) or other smart predicate)
    agent_name: Optional[NameType] = None
    agent_ctx: Optional[str] = None
 
# ============================= Callables ============================ #
# class Callables(BaseModel):
#     """Container for hidden callables parsed from agent instructions."""
#     functions: Dict[HashID, FuncBody] = Field(default_factory=dict)  # key: func HashID
#     apis: Dict[HashID, APIBody] = Field(default_factory=dict)       # key: api HashID

# class ToolConfig(BaseModel):
#     """Configuration for API call."""
#     content:Union[str, None] = None # this is only for in-instruction content.
#     expr_str: str = ""  # the full expression string
#     return_type:str = "str"
#     varnames: List[str] = Field(default_factory=list)

# class BaseTool(BaseModel):
#     hash_id: HashID
    
#     config: ToolConfig
#     # maybe some hints for tool here?


class ToolCard(BaseModel):
    """A brief card for tool representation."""
    name:str = "" # name of the tool group, not the class name



# =============================== Builtins ============================= #
class Builtins(BaseModel):
    """
    Reserved builtin slots resolved from the predicate.
    These are "system-level" hints, not user-defined ordinary params.
    """
    # stage: Stage = Field(default=Stage.BUILD)

    # Instruction template for this agent, usually a quoted string.
    Agent: Optional[QuotedStr] = None

    # IO annotation list, e.g. [+Cond:"...", -Wind:"..."]
    IO: Optional[Dict[VarAnoType, QuotedStr]] = None

    # Model candidates or engine names, e.g. ["gpt-4.1", "deepseek-chat"]
    Model: Optional[List[VarType | QuotedStr]] = None


# ============================== Arguments ============================= #
class Arguments(BaseModel):
    """
    Unified argument slot.
    - 'stage' is derived from the name pattern by add_arg()
    - This struct itself does not guess; it just stores the result.

    About 'ref' and 'hint':
    """
    raw: Any = None
    in_context: bool = False            # whether this arg is from agent context
    type: Optional[ArgTypeHint] = None
    hint: Optional[str] = None          # human/LLM-facing description
    ref: Optional[str] = None           # internal placeholder, e.g. "__VAR__"


# ========================= IRT Parameter Dict ========================= #
# Internal Representation of Intermediate Parameters Dict
# OF A SINGLE LANGDA PREDICATE!!!
class IRTParameterDict(BaseModel):
    """
    Canonical internal representation for one predicate's parameter setting.

    - initials: structural/meta info from parsing
    - callables: resolved function/api configs, keyed by HashID
    - builtins: reserved builtin slots (_Agent, _IO, _Model, ...)
    - args: unified argument table, keys are param names (str),
            values are Arguments with stage derived from naming convention.
    """

    initials: Initials = Field(default_factory=Initials)
    builtins: Builtins = Field(default_factory=Builtins)

    build_basetools: Dict[HashID, ToolCards] = Field(default_factory=dict)
    runtime_basetools: Dict[HashID, ToolCards] = Field(default_factory=dict)
    
    build_args: Dict[BuildVarType, Arguments] = Field(default_factory=dict)
    runtime_args: Dict[RTVarType, Arguments] = Field(default_factory=dict)

    # ----------------------- Public: add argument ----------------------- #
    def add_builtin(self, key: str, value: Any) -> None:
        """
        Register a builtin argument.
        Simply sets the attribute on the builtins model.
        """
        if key not in Builtins.model_fields:
            raise ValueError(f"Invalid builtin param name '{key}'.")
        if getattr(self.builtins, key, None) is not None:
            raise ValueError(f"Builtin '{key}' already assigned.")
        
        self.builtins = self.builtins.model_copy(update={key: value})

    def args_update(self, key: str, body:dict|Arguments) -> None:
        """
        Register an argument.
        The stage (BUILD/RUNTIME) is inferred from the key's pattern:
        - BuildVarType => BUILD
        - RTVarType    => RUNTIME
        Any other pattern => error.
        """
        # Create Arguments instance from body
        if is_valid(BuildVarTypeAdapter, key):
            target_dict = self.build_args
        elif is_valid(RTVarTypeAdapter, key):
            target_dict = self.runtime_args
        else:
            raise ValueError(f"Argument name '{key}' does not match BUILD or RUNTIME patterns.")
        
        # Check for duplicates
        if key in target_dict:
            warnings.warn(
                f"Argument '{target_dict[key]}' already exists and will be overwritten.",
                UserWarning,
            )
        # Add to the appropriate dict
        if not isinstance(body, Arguments):
            body = Arguments(**body)
        target_dict[key] = body

    def tools_update(self, hash_key:str, tool_body: BaseTool, *, has_rtvar:bool = False) -> None:
        """
        Update callables for the given stage.
        has_rtvar: whether the callables contain runtime variables.
        If has_rtvar is True, the callables are for RUNTIME stage; otherwise BUILD stage.
        """
        if has_rtvar:
            target_dict = self.runtime_basetools
        elif not has_rtvar:
            target_dict = self.build_basetools
        else:
            raise ValueError("Cannot determine stage for callables update.")
        # Update functions
        if hash_key in target_dict:
            warnings.warn(
                f"Tool with hash_id '{hash_key}' already exists and will be overwritten.",
                UserWarning,
            )
        target_dict[hash_key] = tool_body
        # Update APIs

    # ---------------------- Public: update metadata --------------------- #


    # ---------------------- Convenience helpers ------------------------ #
    def all_build_args(self) -> Dict[str, Arguments]:
        """Return all BUILD-stage arguments."""
        return {k: v for k, v in self.build_args.items()}

    def all_build_raws(self) -> Dict[str, QuotedStr]:
        """Return all BUILD-stage argument raws."""
        return {k: v.raw for k, v in self.build_args.items()}

    def all_runtime_args(self) -> Dict[str, Arguments]:
        """Return all RUNTIME-stage arguments."""
        return {k: v for k, v in self.runtime_args.items()}
