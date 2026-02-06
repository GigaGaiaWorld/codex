
# ==================== Allowed AST Nodes For Hidden Functions ===================== #
# ------------------------- content.py use these settings ------------------------- #
ALLOWED_NODES_BASE = (
    ast.Expression, ast.Load, ast.Constant, ast.Name,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.UAdd, ast.USub, ast.Not,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, # allow comparisons
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.IfExp, # allow ternary expressions: a if cond else b
    ast.Tuple, ast.List  # allow tuple and list literals: (a, b, c), [a, b, c]
)
# ===================== Safe Functions For Hidden Functions ===================== #
SAFE_FUNC_DEFAULTS: Dict[str, Callable] = {
    # Basic math functions
    'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum,
    # Add more safe functions as needed  
}

"""
class Initials(BaseModel):
    ""
    Parsed structural information from the predicate:
    - stable meta about head and agent context
    ""
    # stage: Stage = Field(default=Stage.INIT)

    hash_id: Optional[HashID] = None

    head_name: Optional[NameType] = None
    head_args: List[VarType | NameType | QuotedStr] = Field(default_factory=list)
    head_ctx: Optional[str] = None

    # Agent source info (e.g. langda(...) or other smart predicate)
    agent_name: Optional[NameType] = None
    agent_ctx: Optional[str] = None


class ToolCard(BaseModel):
    ""A brief card for tool representation.""
    name:str = "" # name of the tool group, not the class name


# =============================== Builtins ============================= #
class Builtins(BaseModel):
    ""
    Reserved builtin slots resolved from the predicate.
    These are "system-level" hints, not user-defined ordinary params.
    ""
    # stage: Stage = Field(default=Stage.BUILD)

    # Instruction template for this agent, usually a quoted string.
    Agent: Optional[QuotedStr] = None

    # IO annotation list, e.g. [+Cond:"...", -Wind:"..."]
    IO: Optional[Dict[VarAnoType, QuotedStr]] = None

    # Model candidates or engine names, e.g. ["gpt-4.1", "deepseek-chat"]
    Model: Optional[List[VarType | QuotedStr]] = None


# ============================== Arguments ============================= #
class TermType(Enum):
    "Types of terms inside predicate/agent arguments."
    POS = "positional" # positional plane argument
    KV = "key_value" # key:value pair argument

@dataclass(frozen=True)
class ArgTerm:
    "A term inside predicate/agent arguments."
    key:str
    value:Optional[str] = None
    term_type:TermType = TermType.POS

    def to_dict(self) -> dict:
        "Convert the ArgTerm to a dictionary representation."
        if self.value is not None: 
            return {self.key: self.value}
        else:
            raise ValueError("Cannot convert ArgTerm to dict without value.")

# Argument Slot
class ToolRaw(BaseModel):
    "Frozen read-only tool callable structure."
    classname: str
    content: str
    args: List[ArgTerm] | None = None
    body: str | None = None
    returns: List[str] | None = None


class Arguments(BaseModel):
    "
    Unified argument slot.
    - 'stage' is derived from the name pattern by add_arg()
    - This struct itself does not guess; it just stores the result.

    About 'ref' and 'hint':
    "
    raw: Optional[Any] = None
    hint: Optional[str] = None          # human/LLM-facing description
    tool_meta: Optional[ToolRaw] = None  # extra tool metadata

# ========================= IRT Parameter Dict ========================= #
# Internal Representation of Intermediate Parameters Dict
# OF A SINGLE LANGDA PREDICATE!!!
class IRTParameterDict(BaseModel):
    "
    Canonical internal representation for one predicate's parameter setting.

    - initials: structural/meta info from parsing
    - callables: resolved function/api configs, keyed by HashID
    - builtins: reserved builtin slots (_Agent, _IO, _Model, ...)
    - args: unified argument table, keys are param names (str),
            values are Arguments with stage derived from naming convention.
    "

    initials: Initials = Field(default_factory=Initials)
    builtins: Builtins = Field(default_factory=Builtins)

    # Callables for BUILD and RUNTIME stages:
    tools: ToolCard = Field(default_factory=ToolCard)

    # Separate arg tables for BUILD and RUNTIME stages:
    # Both tools and vars are stored as "args" here.
    # form: VarName or ToolHead as key, instruction as value
    hidden_args: Dict[HashID, Arguments] = Field(default_factory=dict)
    # hidden_tools: Dict[HashID, Arguments] = Field(default_factory=dict)

    build_args: Dict[HashID, Arguments] = Field(default_factory=dict)
    # build_tools: Dict[BuildToolType, Arguments] = Field(default_factory=dict)

    runtime_args: Dict[HashID, Arguments] = Field(default_factory=dict)
    # runtime_tools: Dict[RTToolType, Arguments] = Field(default_factory=dict)


"""


SCHEMA = {
    # ======================== Hidden inner special parameters  ======================== #
    # ======= These parameters are static once compiled, and can not be changed. ======= #
    # Hidden inner special parameters, if not defined, use default values
    "initials": {
        "hash_id": "auto-generated hash id string",

        "head_name": "weather", # if AllowHead is False, this will be None; could be defined in agent body
        "head_args": ["Cond","Wind"], # if AllowHead is False, this will be empty list
        "head_ctx": "weather(Cond,Wind)",

        "agent_name": "agent name string",
        "agent_ctx": "agent(...)",
    },
    # Hidden inner special function parameters, they will be created automatically when parsing.
    # they are NOT accessible after defined in source code.
    "builtins": {
        # Instruction template for this agent, usually a quoted string.
        "Agent": "instruction string",
        # IO annotation list, e.g. [+Cond:"...", -Wind:"..."]
        "IO": {"+Cond":"the weather condition (e.g., sunny, rainy, cloudy)",
                "-Wind":"str"},
        # Model candidates or engine names, e.g. ["gpt-4.1", "deepseek-chat"]
        "Model": ["gpt-4", "gpt-3.5-turbo"],
    },
    # ======================== Agent Arguments ======================== #
    # Define parameters that will always appear/completed in agent(...)
    # All agents will have these two terms by default:
    "tools": {
        "name": "tool group name string",
    },

    # Choosable agent arguments, could be dynamic or lazy parameters:
    "build_args": {
        # ====================== Speical Dynamic/Lazy Parameters ==================== #
        "HashID": {
            "raw": "quoted string or other raw content",
            "hint": "human/LLM-facing description",
            "tool_meta": {
                "classname": "tool class name string",
                "content": "tool content string",
                "args": [{"key":"arg1", "value":"value1", "term_type":"positional"},
                         {"key":"arg2", "value":"value2", "term_type":"key_value"}],
                "body": "tool body string",
                "returns": ["return1", "return2"],
            }
        },
    },

    "runtime_args": {
        # ================= Regular Dynamic/Lazy Parameters ==================== #
        # Define parameters that will always appear/completed in agent(...)
        "HashID": {
            "raw": "quoted string or other raw content",
            "hint": "human/LLM-facing description",
            "tool_meta": {
                "classname": "tool class name string",
                "content": "tool content string",
                "args": [{"key":"arg1", "value":"value1", "term_type":"positional"},
                         {"key":"arg2", "value":"value2", "term_type":"key_value"}],
                "body": "tool body string",
                "returns": ["return1", "return2"],
            }
        },
    }
}
 
# ===================== BLUEPRINTS {{PLEAS NOTICE: this part is only for demostration for now}} ===================== #
# The schema defines the structure of the parsed predicates and agents.

# There are three stage: 
# 1. source code: src
# 2. graph creation: create_graph()
# 3. runtime execution: runtime()
SCHEMA = {

    # ======================== Hidden inner special parameters  ======================== #
    # ======= These parameters are static once compiled, and can not be changed. ======= #
    # Hidden inner special parameters, if not defined, use default values
    # "__AllowHead__": True,
    "__HASHID__":"auto-generated hash id string",

    "__HeadName__": "weather", # if AllowHead is False, this will be None; could be defined in agent body
    "__HeadArgs__": ["Cond","Wind"], # if AllowHead is False, this will be empty list
    "__HeadCtx__":"weather(Cond,Wind)",


    # Hidden inner special function parameters, they will be created automatically when parsing.
    # they are NOT accessible after defined in source code.
    # use as: __FUNC(Temp * 9/5 + 32 => int)__ , the compute temp * 9/5 + 32, the result is in int form
    # allow usage of default python functions like int(), str(), float(), list(), dict(), sum(), len(), etc.
    # "__FUNC__": {
    #     "func-85b3a410":{"expr_str":"Temp * 9/5 + 32", "return_type":"int", "varnames":["Temp"]},
    #     "func-85b3a410":{"expr":"Temp * 9/5 + 32", "return_type":"int", "varnames":["Temp"]},
    #     # ...
    # },

    # Hidden inner special API parameters, you can only call, not define.
    # use as: __API__(Url)__ or __API__("http://api.weather.com/v3/wx/conditions/current"), the result is a string
    # "__API__": {
    #     "api-12345678":{"endpoint":"http://api.weather.com/v3/wx/conditions/current", "return_type":"str"},
    #     # ...
    # },

    # you could also define your own hidden parameters here,
    # they can not be touched in the source code, and always use default values.
    # "__Docs__": "Some hidden documentation string for internal use?",


    # ======================== Agent Arguments ======================== #
    # Define parameters that will always appear/completed in agent(...)
    # All agents will have these two terms by default:
    "__AgentName__":"agent name string",
    "__AgentCtx__":"agent(...)",

    # Choosable agent arguments, could be dynamic or lazy parameters:
    # Hidden inner special parameters, if not defined, use default type (Allowing delayed definition in source code)
    "__Args__":{ 
        # "__Name__": "weather", # define name if no head
        # ====================== Speical Dynamic/Lazy Parameters ==================== #
        # Agent:"...", instruction string for this agent
        "_Agent": str, 
        # IO: [+Cond:"weather good or bad?", +Wind:"wind speed in m/s"],
        "_IO": {
            "+Cond":"the weather condition (e.g., sunny, rainy, cloudy)",
            "-Wind":"str",
            "Temp_C":float,
        },
        # special lazy parameter: model, if not defined in source code, use default model as framework defined
        "_Model": "gpt-4", # lazy parameter


        # ================= Regular Dynamic/Lazy Parameters ==================== #
        # Lazy parameters: could define type only in src, but must be defined as input in create_graph(),
        # Dynamic parameters: could define type only in src and create_graph(), but must be defined as input in runtime()
        # Define parameters that will always appear/completed in agent(...)

        "WeatherAPI": str, # dynamic parameter
        "SensorData": List[float], # dynamic parameter,
        "_EmployeeInfo": "Info: Mr. Smith, ID: 12345, Dept: Sales", # lazy parameter
        }

}
SCHEMA = {
    "agent_name": "langda",
    "args_schema": {
        "Model": "model mame string",
        "Agent": str,

        "Meta": dict,

    }
}

