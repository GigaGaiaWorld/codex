from __future__ import annotations
import re, json, hashlib
from functools import partial
from dataclasses import dataclass, field
from typing import Tuple, Union, Literal, List, Annotated, Any, get_args, Optional
from pydantic import TypeAdapter, StringConstraints, BeforeValidator, ValidationError
from .schemas import PlaceholderStyle, _DELIMS
"""
IN case of: 
1. weather(Cond,Wind)@agent(...)@llm(...).
2. langda(...).
"""
 
_HASH_LEN: int = 8 # fixed length
_HEX_RE = rf'^[A-Za-z][A-Za-z0-9_]*_[0-9a-f]{{{_HASH_LEN}}}$'
_DIGEST_LEN = {
    "md5": 32,
    "blake2s": 64,
}

# ================= Frozen read-only packet (fixed structure) ================= #
Span = Tuple[int, int]

@dataclass(slots=True)
class Pack: # for head and following agent terms
    name:str # agent or langda
    args:str # (...)

    @staticmethod
    def parse(raw:str) -> Pack:
        if not isinstance(raw, str):
            try:
                raw = str(raw) or raw.value
            except:
                raise ValueError(f"Cannot convert {raw} to Pack")
        # split from the first '(', the first part is name, the rest is args
        name, sep, rest = raw.partition('(')
        if not sep:
            return Pack(name=raw.strip(), args="")
        if not sep or not rest.endswith(')'):
            raise ValueError(f"Invalid Pack format: {raw}")
        return Pack(name=name.strip(), args=sep+rest.strip())

@dataclass(slots=True)
class PredRaw:
    head:Pack|None = None # weather(Cond,Wind) or none
    agents:List[Pack] = field(default_factory=list)

    @staticmethod
    def dump(head:Optional[str], agent:Optional[str]) -> str:
        return PredRaw(
            head=Pack.parse(head) if head else None,
            agents=[Pack.parse(agent)] if agent else []
        )
    def __repr__(self):
        head = f"{self.head.name}{self.head.args}" if self.head else ""
        agents = ', '.join([f"{agent.name}{agent.args}" for agent in self.agents])
        return f"{head}({agents})"

# ================ Builder (responsible for consistency and freezing) ========== #
# File: src/parser/recognizer.py will use these builders to build predicates.
# temp_headname = SpanDraft() => temp_headname.open(...),temp_headname.close(...) => 
# temp_pred = PredBuilder() => temp_pred.set_head(temp_headname.as_span(),temp_headargs.as_span())
class IncompleteSpanError(Exception):
    """Raised when trying to freeze an incomplete predicate."""
    def __init__(self, span:Span):
        # this will be passed to upper level for debugging, 
        # for example: ctx.src(span[0]:span[1]) => real text
        self.span = span 
        super().__init__(f"Incomplete span cannot be frozen: {span}")

@dataclass(slots=True)
class SpanDraft: # Used for building spans step by step, whenever you want to create a span, use this.
    start:int|None = None
    end:int|None = None

    def is_none(self) -> bool:
        return self.start is None and self.end is None
    def is_open(self) -> bool:
        return (self.start is None) != (self.end is None)
    def is_complete(self) -> bool:
        return self.start is not None and self.end is not None

    def open(self, start:int) -> None: self.start = start # left
    def close(self, end:int) -> None: self.end = end      # right
    def dump(self, *, reset:bool = True) -> Span:
        if self.start is None or self.end is None:
            raise IncompleteSpanError((self.start, self.end))
        span = (self.start, self.end)
        if reset:
            self.start, self.end = None, None
        return span


# ============================= Type Adapters ============================== #
# Name type with validation: starts with letter or underscore, followed by letters, digits, or underscores.
# for example: valid names are "Weather", "my_Var1", "__hidden", for head/agent names, lazy/dynamic parameter names
NameType = Annotated[str, StringConstraints(min_length=1, pattern=r'^[A-Za-z_][A-Za-z0-9_]*$')]
NameTypeAdapter = TypeAdapter(NameType)

# Argument type with validation: starts with upper letter, then letters, digits, or underscores.
# for example: valid argument names are "Result", "Output_Var", "Data1"
VarType = Annotated[str, StringConstraints(min_length=1, pattern=r'^(?:_?[A-Z][A-Za-z0-9_]*|_)$')]
VarTypeAdapter = TypeAdapter(VarType)# use validator to check

BuildVarType = Annotated[str, StringConstraints(min_length=1, pattern=r'^(?:_[A-Z][A-Za-z0-9_]*|_)$')]
BuildVarTypeAdapter = TypeAdapter(BuildVarType)

RTVarType = Annotated[str, StringConstraints(min_length=1, pattern=r'^(?:[A-Z][A-Za-z0-9_]*|_)$')]
RTVarTypeAdapter = TypeAdapter(RTVarType)

# Variable annotation type: starts with optional +/-, followed by upper letter, then letters, digits, or underscores.
# for example: valid variable annotations are "+Result", "-Output_Var", "Data1"
VarAnoType = Annotated[str, StringConstraints(min_length=1, pattern=r'^[+-]?[A-Z][A-Za-z0-9_]*$')]
VarAnoTypeAdapter = TypeAdapter(VarAnoType) # use validator to check

# Quoted string type: enclosed in single or double quotes
def quoted_str_validator(v: str) -> str:
    """Validate that the string is enclosed in single or double quotes."""
    if not isinstance(v, str):
        raise TypeError("Value must be a string")
    if not re.fullmatch(r"(['\"])(.*)\1", v):
        raise ValueError("String must be enclosed in single or double quotes")
    return v[1:-1]

def _unquote_functor(f: Any) -> str:
    """ProbLog binops often become "'op'" (including quotes). Normalize to op."""
    s = str(f)
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1]
    return s

QuotedStr = Annotated[str, BeforeValidator(quoted_str_validator)]
QuotedStrTypeAdapter = TypeAdapter(QuotedStr)

# NameCall type: in the form Name(...) => tool head
def name_call_validator(v: str, *, var_adapter:TypeAdapter) -> Tuple[str, str]:
    m = re.fullmatch(
        r'^(?P<name>.+?)\((?P<body>.*)\)$',  # 先粗暴拆成 name 和 body
        v.strip(),
    )
    if not m:
        raise ValueError('String must be in form "Name(...)"')
    name = m.group('name')
    body = m.group('body')
    try:
        var_adapter.validate_python(name)
    except Exception as e:
        raise ValueError(f"Invalid name part in NameCall: {name!r} ({e})")
    return name, body

BuildToolType = Annotated[Tuple[str,str], BeforeValidator(partial(name_call_validator, var_adapter=BuildVarTypeAdapter))]
BuildToolTypeAdapter = TypeAdapter(BuildToolType)
RTToolType = Annotated[Tuple[str,str], BeforeValidator(partial(name_call_validator, var_adapter=RTVarTypeAdapter))]
RTToolTypeAdapter = TypeAdapter(RTToolType)
ToolType = Annotated[Tuple[str,str], BeforeValidator(partial(name_call_validator, var_adapter=VarTypeAdapter))]
ToolTypeAdapter = TypeAdapter(ToolType)


# List term handler: validate and clean list terms like [...]
def list_bracket_validator(v):
    """Validate that the value is a string enclosed in [ ]."""
    if not isinstance(v, str):
        raise TypeError("Value must be a string")
    if not (len(v) >= 2 and v[0] == "[" and v[-1] == "]"):
        raise ValueError("String must start with '[' and end with ']'")
    return v
ListType = Annotated[str, BeforeValidator(list_bracket_validator)]
ListTypeAdapter = TypeAdapter(ListType)

### Helper function to check validity using TypeAdapter:
# <--------------> Validity Checkers <--------------> #
def is_valid(adapter: TypeAdapter, v: str) -> bool:
    try:
        adapter.validate_python(v)
        return True
    except (ValidationError, TypeError, ValueError):
        return False  

def get_pattern(in_type: Any) -> str:
    """Get regex pattern from Annotated type with StringConstraints."""
    _, constraint = get_args(in_type)
    if isinstance(constraint, StringConstraints) and constraint.pattern is not None:
        return constraint.pattern
    raise ValueError("No pattern found in the provided Annotated type.")


# -------------------- Name Conversion Utilities -------------------- #
def to_dsl_name(py_name: str) -> str:
    """Convert a Pythonic name (snake_case) to DSL name (CamelCase)."""
    parts = py_name.split('_')
    return ''.join(word.capitalize() for word in parts)



def to_pythonic_name(name:str) -> str:
    """Convert DSL style name to Pythonic style name."""
    try:
        if not name:
            return name
        for i, c in enumerate(name):
            if c.isupper():
                if i == 0:
                    py_name = c.lower()
                else:
                    py_name += "_" + c.lower()
            else:
                py_name += c
        return py_name
    except:
        raise ValueError(f"Cannot convert name '{name}' to pythonic style.")



# ============================== Extractor ============================== #
# File: src/parser/predicate_div.py will use this Extractor to extract hashid and placeholder.
# Other files may also use it to generate placeholder from hash.

# Extract Hash ID and formatted placeholder from content.
# Different hashing algorithms and placeholder styles are supported.
# Call the render() method to get HashID and placeholder


# Define HashID type with validation
HashID = Annotated[str, StringConstraints(pattern=_HEX_RE)]
HashIDAdapter = TypeAdapter(HashID)

class Extractor:
    """
    Extract Hash ID and formatted placeholder from content.
    Different hashing algorithms and placeholder styles are supported.
    Call the render() method to get HashID and placeholder,
    Call the placeholder() method to get formatted placeholder from hashid.
    Returns: hash_id: HashID, placeholder: str

    ====> only predefine the placeholder style and hashing algorithm,
        the prefix will be provided during render() call. <====
    """
    def __init__(self, 
                 algo:Literal["blake2s","md5"],
                 placeholder_style:PlaceholderStyle,
                 *,
                 custom_front:str|None = None,
                 custom_end:str|None = None
    ):
        self.algo:Literal["blake2s","md5"] = algo
        self.placeholder_style:PlaceholderStyle = placeholder_style
        self.custom_front:str = custom_front
        self.custom_end:str = custom_end
        # Validate hash_len:
        max_len = _DIGEST_LEN[self.algo]
        if not (1 <= _HASH_LEN <= max_len):
            raise ValueError(f"hash_len must be in [1, {max_len}] for {self.algo}")

    @staticmethod
    def _normalize_content(content:Union[str, dict]) -> str:
        """Normalize content from or dict for hashing.""" 
        if isinstance(content, dict):
            # For dict, serialize to JSON with sorted keys, ensure consistent representation.
            return json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return str(content).encode("utf-8")

    def _create_hex_digest(self, bytes: bytes) -> str:
        """Create hex digest based on the selected algorithm from bytes."""
        if self.algo == "blake2s": # blake2s produces up to 64 hex chars
            return hashlib.blake2s(bytes).hexdigest()
        elif self.algo == "md5": # default, md5 produces 32 hex chars
            return hashlib.md5(bytes).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algo: {self.algo}")

    def _create_hashid(self, content:str, *, prefix:str) -> HashID:
        """Create Hash ID.
        1. Computes a short MD5 hash of the given content (str or dict) with specified length.
        2. Returns hash ID with optional prefix.
        """
        if not prefix.isalpha():
            raise ValueError("prefix must be letters only (A-Z or a-z)")
        text = self._normalize_content(content)
        digest = self._create_hex_digest(text)
        out_hash_id = f"{prefix}_{digest[:_HASH_LEN]}" # create hash id in format: prefix-xxxxxxxx
        return HashIDAdapter.validate_python(out_hash_id) # validate format

    def placeholder(self, raw:str, *, as_pattern = False) -> str:
        """Return a formatted placeholder string according to the chosen style."""
        if self.placeholder_style == PlaceholderStyle.CUSTOM:
            if self.custom_front is None or self.custom_end is None:
                raise ValueError("custom_front and custom_end must be provided for CUSTOM style.")
            left, right = self.custom_front, self.custom_end
        elif self.placeholder_style in _DELIMS:
            left, right = _DELIMS[self.placeholder_style]
        else:
            raise ValueError(f"Invalid placeholder style: {self.placeholder_style}")
        if as_pattern:
            left = re.escape(left)
            right = re.escape(right)
            inner = raw.lstrip('^').rstrip('$')  # remove anchors if present
            return f"{left}\s*({inner})\s*{right}"
        return f"{left}{raw}{right}"
    
    def render(self, content:str, *, prefix:str) -> Tuple[HashID, str]:
        """Return a formatted placeholder string according to the chosen style."""
        hash_id:HashID = self._create_hashid(content, prefix=prefix)
        placeholder = self.placeholder(hash_id)
        return hash_id, placeholder