from enum import Enum
from typing import Dict, Callable

# ========================= Allowed Brackets and Depths ========================== #
# ----------------------- predicate.py use these settings ------------------------ #
# _ALLOWED_BRACKETS = {'(':')', '}':'}', '[':']'}
_ALLOWED_BRS = {'[':']','(':')'} # currently only list brackets are supported
_ALLOWED_DPS = 1 # maximum depth of nested brackets allowed inside agent arguments

class _SPECIAL_TERMS(Enum):
    NAME = "Name"
    IO = "IO"
    AGENT = "Agent"
    MODELS = "Models"

class PlaceholderStyle(Enum):
    CURLY = "curly"
    SQUARE = "square"
    CUSTOM = "custom"

_DELIMS = { # predefined delimiters
    PlaceholderStyle.CURLY: ("{{", "}}"),
    PlaceholderStyle.SQUARE: ("[[", "]]"),
}

# ===================== Allowed Return Types For Hidden Callables ==================== #
_ALLOWED_RETURN_TYPES: Dict[str, Callable] = {
    "int":int,
    "float":float,
    "str":str,
    "bool":bool,
    "json":str, # keep json as str, let user parse it later
}
