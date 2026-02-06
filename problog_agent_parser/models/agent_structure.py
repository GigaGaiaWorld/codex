from __future__ import annotations
from typing import Any, List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from .models import (
    HashID, NameType, VarType, QuotedStr, VarAnoType,
    QuotedStrTypeAdapter, HashIDAdapter, BuildVarTypeAdapter,
    RTVarTypeAdapter, BuildToolTypeAdapter, RTToolTypeAdapter,
    is_valid, to_dsl_name,
)
from problog.logic import Term, Clause

import warnings
"""
This module defines core data structures for predicates, including:
    - Initials for parsed structural info
    - Builtins for reserved builtin slots
    - Arguments for unified argument representation
They will be composed into IRTDict (Internal Representation of Intermediate Parameters Dict),
which serves as the canonical internal representation of predicate parameters.

This Intermediate Representation (IR) is expected to be used '''across all stages''', 
maybe we will put it elsewhere in the future when we agentic the agent parts.

Some notes on the design:
1) We separate 'agentic' and 'runtime' arguments explicitly in IRTDict,
"""

# ============================= Stage Enum ============================= #
# class Stage(str, Enum):
#     INIT = "init"        # layer0: parsing/meta
#     BUILD = "agentic"      # layer1: graph construction
#     RUNTIME = "runtime"  # layer2: runtime execution

# =============================== Initials ============================= #
class Initials(BaseModel):
    """
    Parsed structural information from the predicate:
    - stable meta about head and agent context
    """

    head_name: Optional[NameType] = None
    head_args: List[str] = Field(default_factory=list) # 
    has_unknown_args:bool = False

    # Agent source info (e.g. langda(...) or other smart predicate)
    agent_name: Optional[NameType] = None
    # This links to the path inside AST where this agent is defined
    # we need this because we treat agents as first-class citizens
    anchor_stmt_idx: Optional[int] = None  # statement index in program
    anchor_path: Optional[Any] = None  # path to args inside ast

# =============================== Builtins ============================= #
class Builtins(BaseModel):
    """
    Reserved builtin slots resolved from the predicate.
    These are "system-level" hints, not user-defined ordinary params.
    """
    # Name of head:
    Name: Optional[QuotedStr] = None  # The head name as a quoted string.
    # Instruction template for this agent, usually a quoted string.
    Agent: Optional[QuotedStr] = None
    # IO annotation list, e.g. [+Cond:"...", -Wind:"..."]
    IO: Optional[Dict[VarAnoType, QuotedStr]] = None
    # Model candidates or engine names, e.g. ["gpt-4.1", "deepseek-chat"]
    Models: Optional[List[VarType | QuotedStr]] = None
    # Information for user in yaml file to distinguish different agent blocks
    Info: Optional[QuotedStr] = Field(default="No Additional Info Provided.")

# ============================== Tool Arguments ============================= #
@dataclass(frozen=True)
class ArgTerm:
    """A term inside predicate/agent arguments."""
    key:str
    value:Optional[str] = None
    # whether this term is a transit term, if true, value points to another param
    transit:bool = False  

# Argument Slot
class ToolConfig(BaseModel):
    """Frozen read-only tool callable structure."""
    classname: str
    type: Literal["Instant","Agentic","Runtime"] = "Instant"
    content: str
    args: List[ArgTerm] | None = None
    mode: str | None = None
    returns: List[str] | None = None

class Arguments(BaseModel):
    """
    Unified argument representation for predicate/tool parameters.
    - raw: original raw string from parsing
    - hint: human/LLM-facing description
    - conf: extra tool metadata if this argument is a tool
    =====》 still thinking about should every arg have conf or not
    """
    hash_id: Optional[HashID] = None
    raw: Optional[Any] = None
    hint: Optional[str] = None          # human/LLM-facing description
    conf: Optional[ToolConfig] = None  # extra tool metadata


# ========================= IRT Parameter Dict ========================= #
# Internal Representation of Intermediate Parameters Dict MAP
class IRTMap(dict[HashID, 'IRTDict']):
    """Mapping from HashID to Arguments for all predicates in a scenario."""
    def add_dict(self, ir_dict: 'IRTDict') -> None:
        """Add an IRTDict to the map."""
        if ir_dict.hash_id is None:
            raise ValueError("Cannot add IRTDict without hash_id to IRTMap.")
        if ir_dict.hash_id in self:
            warnings.warn(
                f"IRTDict with hash_id '{ir_dict.hash_id}' already exists and will be overwritten.",
                UserWarning,
            )
            return
        self[ir_dict.hash_id] = ir_dict

    def get_dict(self, hash_id: HashID) -> Optional['IRTDict']:
        """Get an IRTDict by hash_id."""
        return self.get(hash_id, None)

# ============================ Candidate Structs ============================ #
@dataclass
class Candidate:
    """
    A structure to hold the transit head and candidate terms for a specific agent.
    - transit_head: The intermediate term used in the probabilistic logic.
    - candidate_terms: A list of clauses representing different candidate definitions.
    """
    transit_head: Term # agent_hash_id(...terms...)
    primary_key: str
    candidate_terms: List[Clause] = field(default_factory=list)

# ========================= IRT Parameter Dict ========================= #
# Internal Representation of Intermediate Parameters Dict
# OF A SINGLE LANGDA PREDICATE!!!
class IRTDict(BaseModel):
    """
    Canonical internal representation for one predicate's parameter setting.

    - initials: structural/meta info from parsing
    - callables: resolved function/api configs, keyed by HashID
    - builtins: reserved builtin slots (_Agent, _IO, _Model, ...)
    - args: unified argument table, keys are param names (str),
            values are Arguments with stage derived from naming convention.
    """
    hash_id: Optional[HashID] = None
    placeholder: Optional[bool] = None

    initials: Initials = Field(default_factory=Initials)
    builtins: Builtins = Field(default_factory=Builtins)
    candidate: Optional[Candidate] = None

    # Separate arg tables for BUILD and RUNTIME stages:
    # Both tools and vars are stored as "args" here.
    # form: VarName or ToolHead as key, instruction as value
    instant_args: Dict[HashID, Arguments] = Field(default_factory=dict)
    # instant_tools: Dict[HashID, Arguments] = Field(default_factory=dict)

    agentic_args: Dict[HashID, Arguments] = Field(default_factory=dict)
    # agentic_tools: Dict[BuildToolType, Arguments] = Field(default_factory=dict)

    runtime_args: Dict[HashID, Arguments] = Field(default_factory=dict)
    # runtime_tools: Dict[RTToolType, Arguments] = Field(default_factory=dict)

    # ----------------------- Public: add argument ----------------------- #
    def __repr__(self) -> str:
        return f"IRTDict(hash_id={self.hash_id})"

    def get_head_ctx(self) -> str:
        """Get a string representation of the head with its arguments."""
        return f"{self.initials.head_name}({', '.join([arg for arg in self.initials.head_args])})"

    def builtins_update(self, key: str, value: Any, force:bool=False) -> None:
        """
        Register a builtin argument.
        Simply sets the attribute on the builtins model.
        """
        if key not in Builtins.model_fields:
            raise ValueError(f"Invalid builtin param name '{key}'.")
        if getattr(self.builtins, key, None) is not None and not force:
            raise ValueError(f"Builtin '{key}' already assigned.")
        
        self.builtins.__setattr__(key, value)


    def add_args(self, hash_id:str, key:str, hint:str) -> None:
        """
        Register an argument.
        The stage (BUILD/RUNTIME) is inferred from the key's pattern:
        - BuildVarType|BuildToolType => BUILD
        - RTVarType|RTToolType    => RUNTIME
        Any other pattern => error.
        """
        if not is_valid(HashIDAdapter, hash_id):
            raise ValueError(f"Argument hash_id '{hash_id}' is not a valid HashID.")

        if is_valid(BuildVarTypeAdapter, key) or is_valid(BuildToolTypeAdapter, key):
            target_dict = self.agentic_args
        elif is_valid(RTVarTypeAdapter, key) or is_valid(RTToolTypeAdapter, key):
            target_dict = self.runtime_args
        else:
            raise ValueError(f"Argument name '{key}' does not match BUILD or RUNTIME patterns.")
        if hint is not None and (not is_valid(QuotedStrTypeAdapter, hint)):
            raise ValueError(f"Argument hint '{hint}' of key {key} must be a quoted string.")
        body = Arguments(
            hash_id=hash_id,
            raw=key,
            hint=hint.strip("'").strip('"') if hint else None
        )
        # Check for duplicates
        if hash_id in target_dict:
            warnings.warn(
                f"Argument '{key}' already existing terms will be overwritten.",
                UserWarning,
            )
        # Add to the appropriate dict
        target_dict[hash_id] = body

    def update_args(self, hash_id: str, body:dict|Arguments) -> None:
        """
        Update an existing argument.
        The stage (BUILD/RUNTIME) is inferred from the name's pattern:
        - BuildVarType|BuildToolType => BUILD
        - RTVarType|RTToolType    => RUNTIME
        Any other pattern => error.
        """
        is_tool = True
        if body.conf is None:
            is_tool = False
        name = body.raw if not is_tool else body.conf.classname

        if is_valid(BuildVarTypeAdapter, name) or is_valid(BuildToolTypeAdapter, name):
            tool_type = "Agentic"
            target_dict = self.agentic_args
            name = name.split("_")[1]
            body.conf.classname = name

        elif is_valid(RTVarTypeAdapter, name) or is_valid(RTToolTypeAdapter, name):
            tool_type = "Runtime"
            target_dict = self.runtime_args
        else:
            raise ValueError(f"Argument name '{name}' does not match BUILD or RUNTIME patterns.")

        if not isinstance(body, Arguments):
            body = Arguments(**body)
        # Update tool type if applicable
        body.conf.type = tool_type

        # Merge with existing argument if any
        update_data = body.model_dump(exclude_none=True)
        if hash_id in target_dict:
            merged = target_dict[hash_id].model_dump()
            merged.update(update_data)
            target_dict[hash_id] = Arguments(**merged)
        else:
            target_dict[hash_id] = body

    def add_instant_args(self, name: str, body:dict|Arguments) -> None:
        """
        Register a instant tool.
        The stage (BUILD/RUNTIME) is inferred from the name's pattern:
        - HashID => HIDDEN RUNTIME
        Any other pattern => error.
        """
        if is_valid(HashIDAdapter, name):
            target_dict = self.instant_args
        else:
            raise ValueError(f"Hidden argument name '{name}' does not match RUNTIME instant arg/tool patterns.")

        if not isinstance(body, Arguments):
            body = Arguments(**body)
        # Check for duplicates
        if name in target_dict:
            warnings.warn(
                f"Hidden argument/tool '{target_dict[name]}' already exists and will be overwritten.",
                UserWarning,
            )
        # Add to the appropriate dict
        target_dict[name] = body

    # ---------------------- Convenience helpers ------------------------ #
    def all_agentic_args(self) -> Dict[str, Arguments]:
        """Return all BUILD-stage arguments."""
        return {k: v for k, v in self.agentic_args.items()}

    def all_agentic_raws(self) -> Dict[str, QuotedStr]:
        """Return all BUILD-stage argument raws."""
        return {k: v.raw for k, v in self.agentic_args.items()}

    def all_runtime_args(self) -> Dict[str, Arguments]:
        """Return all RUNTIME-stage arguments."""
        return {k: v for k, v in self.runtime_args.items()}

    def all_runtime_raws(self) -> Dict[str, QuotedStr]:
        """Return all RUNTIME-stage argument raws."""
        return {k: v.raw for k, v in self.runtime_args.items()}
    
    def all_agentic_and_runtime_args(self) -> Dict[str, Arguments]:
        """Return all BUILD and RUNTIME-stage arguments."""
        combined = {**self.agentic_args, **self.runtime_args}
        return combined

    def print_irt(self) -> None:
        print(
            self.model_dump_json(
                indent=2,
            )
        )

    def to_block_meta(self) -> dict:
        """Convert IRTDict to block meta dict for yaml serialization."""
        meta_id = self.hash_id if self.hash_id else "unknown_hash"

        if self.initials.head_name:
            head_name = self.initials.head_name
        else:
            head_name = "Not Provided"
        
        if self.builtins.Info:
            info = self.builtins.Info
        else:
            info = "No Additional Info Provided."

        return {
            "id": meta_id,
            "head": head_name,
            "info": info,
        }
    
    def to_block_context_template(self) -> dict:
        """Convert IRTDict to block context template for yaml serialization."""
        create_graph_defaults: Dict[str, Any] = {}
        create_graph_params: Dict[str, Any] = {}
        runtime_defaults: Dict[str, Any] = {}
        runtime_inputs: Dict[str, Any] = {}
        def add_to_params(argument:Arguments, inputs:dict, defaults:dict) -> None:
            if argument.conf is None:
                return
            args = argument.conf.args
            if args is None:
                return
            for arg in args:
                if arg.transit:
                    inputs[arg.value] = ""
                elif arg.value is None:
                    inputs[arg.key] = ""
                else:
                    defaults[arg.key] = arg.value

        for _hash, tool in self.instant_args.items():
            add_to_params(tool, create_graph_params, create_graph_defaults)

        for _hash, arg in self.agentic_args.items():
            add_to_params(arg, create_graph_params, create_graph_defaults)
        
        for _hash, arg in self.runtime_args.items():
            add_to_params(arg, create_graph_params, create_graph_defaults)
            add_to_params(arg, runtime_inputs, runtime_defaults)

        agent_name = to_dsl_name(self.initials.agent_name)
        block_name = f"{agent_name} Block"
        block_section: Dict[str, Any] = {
            block_name: {
                "meta": self.to_block_meta(),
                "create_graph": {
                    "defaults": create_graph_defaults,
                    "inputs": create_graph_params,
                },
                "runtime": {
                    "defaults": runtime_defaults,
                    "inputs": runtime_inputs,
                },
            }
        }
        return block_section