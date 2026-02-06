from typing import List, Optional, Tuple
from problog.program import DefaultPrologFactory
from problog.parser import PrologParser, Token, UnexpectedCharacter
from problog.logic import Term, Var, Constant, And, Or, Not, Clause
from dataclasses import dataclass

from models.state import Ctx, ParserState
from models.models import PredRaw, Extractor
from models.agent_structure import IRTDict, IRTMap
from parser.recognizer import Recognizer, AgentRawRecognizer, ToolRecognizer
from parser.predicate_parser import PredicateParser
from parser.tool_parser import IncontextToolParser
from create_tools.tools_registry import _TOOLKIT_REGISTRY
from create_tools.basetools import *

_AGENT_NAMES = ["agent", "langda"]
_SPECIAL_AGENT_RAW = 114

PathStep = Tuple[str, int]

def _set_term_args_inplace_by_path(body: Any, path: Tuple[PathStep, ...], new_args: Any) -> Term:
    if path and path[0][0] == "clause_body":
        path = path[1:]

    cur = body
    for kind, idx in path:
        if kind == "binop":
            cur = cur.op1 if idx == 0 else cur.op2
        elif kind == "not":
            cur = cur.child
        elif kind == "term_arg":
            cur = cur.args[idx]
        else:
            raise ValueError(f"bad path step: {(kind, idx)}")

    if not isinstance(cur, Term):
        raise TypeError(f"Target at path is not a Term, got: {type(cur)}")

    cur.args = tuple(new_args)
    return cur


def update_agent_call_args(stmts: List[Any], irtmap: IRTMap) -> None:
    """
    In-place 更新所有 agent 调用点的 args。
    默认每个 irt_dict.initials.new_args 都已经准备好。
    这一步操作是比较靠后的, 最后进行整体测试需要的, 前面的测试都是基于单元测试
    示例:
    weather(X,Y) :- head(A)@agent(Agent:"Decide weather", Var(V:"Weather decision")), choose(Color).
    =>
    weather(X,Y) :- agent_hash_id(...new_args...), choose(Color).

    agent_hash_id(...new_args...) :- # Transit head: only call the corresponding predicate, 
        new_head(...new_args...).    # This is generated automatically(not by llm)
    
    new_head(...args...) :- ...
    new_head(...args...) :- ...
    """
    # 遍历所有 agent dicts, 找到锚点, 更新 args:
    for agent_id, irt_dict in irtmap.items():
        stmt_idx = getattr(irt_dict.initials, "anchor_stmt_idx", None)
        path = getattr(irt_dict.initials, "anchor_path", None)
        candidate = getattr(irt_dict.candidate, "transit_head", None)
        new_args = candidate.args # 直接用 candidate 的 args 作为新的 args => agent_hash_id( ...new_args... )

        if stmt_idx is None or path is None:
            continue
        if new_args is None:
            continue

        st = stmts[stmt_idx]
        if not isinstance(st, Clause):
            continue

        _set_term_args_inplace_by_path(st.body, path, new_args)

    # 在后面, 我们还需要把 candidate_terms 也合并到最终代码里去:



def index_agent_paths(stmts: List[Any], irtmap:IRTMap) -> None:
    """
    In-place: 对每个 agent term，把锚点写进 IRTDict：
      - initials.anchor_stmt_idx
      - initials.anchor_path  (Tuple[PathStep,...])
    """

    def walk(e: Any, stmt_idx: int, path: List[PathStep]):
        # 只认带 _hash_id 的 Term（你的 agent 一等公民节点）
        if isinstance(e, Term) and getattr(e, "_hash_id", None) is not None:
            aid = e._hash_id
            irt_dict = irtmap.get_dict(aid)
            # 存 stmt_idx + tuple(path)
            irt_dict.initials.anchor_stmt_idx = stmt_idx
            irt_dict.initials.anchor_path = tuple(path)
            return

        if isinstance(e, (And, Or)):
            walk(e.op1, stmt_idx, path + [("binop", 0)])
            walk(e.op2, stmt_idx, path + [("binop", 1)])
            return

        if isinstance(e, Not):
            walk(e.child, stmt_idx, path + [("not", 0)])
            return

        if isinstance(e, Term):
            for i, a in enumerate(e.args):
                walk(a, stmt_idx, path + [("term_arg", i)])
            return

    for i, st in enumerate(stmts):
        if isinstance(st, Clause):
            walk(st.body, i, [("clause_body", 0)])

    

class AgentPrologFactory(DefaultPrologFactory):
    """
    Custom Prolog Factory to handle Agent '@' operator parsing.
    It overrides build_function to specially process the '@' operator,
    extracting the agent term and parsing it into an IRTDict.
    1) If functor is not '@', use default behavior.
    2) If functor is '@':
        - Extract head and agent arguments.
        - Parse agent argument into PredRaw.
        - Use PredicateParser to parse PredRaw into IRTDict.
        - Use IncontextToolParser to further parse instant tool calls inside agent term.
        - Construct a special Term '__AGENT_@_TERM__' with the IRTDict as argument.
    
    As print, it will produce a Term like:
    __AGENT(irt_dict_hash_id)TERM__
    """
    def __init__(self, 
                 extractor:Extractor,
                 recognizers:Optional[dict] = None,
                 identifier=0
    ):
        self.extractor = extractor
        self.pred_parser = PredicateParser(
            extractor=extractor,
            case_sensitive=True
        )
        self.tool_parser = IncontextToolParser(
            extractor=extractor,
            case_sensitive=True
        )
        self.tool_parser.add_tool_recognizer(recognizers["tool"])
        self._agent_registry = IRTMap()
        DefaultPrologFactory.__init__(self, identifier=identifier)

    def build_function(self, functor, arguments, location=None, **extra):
        """Override to handle '@' operator for agent terms.
        Content: 
        - If functor is not '@', use default behavior.
        - If functor is '@':
            - Extract head and agent arguments.
            - Parse agent argument into PredRaw.
            - Use PredicateParser to parse PredRaw into IRTDict.     
            - Use IncontextToolParser to further parse instant tool calls inside agent term.
            - Construct a special Term '__AGENT_@_TERM__' with the IRT
            Dict as argument.
        Returns:
        - Term: either normal Term or special Agent Term.
        - Special Agent Term has attribute '_hash_id' for later identification.
        """
        if not functor == "'@'":
            return super().build_function(functor, arguments, location, **extra)

        if len(arguments) == 2:
            head, agent = arguments
        elif len(arguments) == 1:
            (agent,) = arguments
            head = None
        agent_predraw = PredRaw.dump(
            head=head,
            agent=agent,
        )
        # parse agent term into IRTDict
        self.pred_parser.reset(agent_predraw)
        ir_dict = self.pred_parser.parse()

        # further parse instant tool calls inside agent term
        self.tool_parser.reset(ir_dict)
        ir_dict = self.tool_parser.parse()

        agent_term = Term(
            ir_dict.hash_id,
            Var("_"),  # placeholder for easy identification
            location,
            **extra,
        )
        # agent_term.repr = ir_dict.initials.placeholder  # for pretty printing
        self._hash_id = ir_dict.hash_id
        self._agent_registry.add_dict(ir_dict) # register the ir_dict

        return agent_term




def assign_recognizers(agent_names:Optional[List[str]] = None) -> dict:
    recognizers = {}
    if agent_names is None:
        agent_names = _AGENT_NAMES
    recognizers["agent_raw"] = AgentRawRecognizer(agent_names, case_sensitive=True)
    recognizers["tool"] = ToolRecognizer(_TOOLKIT_REGISTRY, case_sensitive=True)
    return recognizers

class AgentAtParser(PrologParser):
    def __init__(self,
                 extractor:Optional[Extractor] = None,
                 agent_names:Optional[List[str]] = None,
    ):
        self.recognizers = assign_recognizers(agent_names)
        super().__init__(factory=AgentPrologFactory(extractor, self.recognizers))
        self._pending_tokens = []   # [(Token, newpos)]
        self.agent_parser = SingleAgentSimpleParser(
            case_sensitive = True,
        )
        self.agent_parser.add_recognizer(self.recognizers["agent_raw"])

    def next_token(self, s, pos):
        # Hijack pending tokens first!!!
        if self._pending_tokens:
            tok, newpos = self._pending_tokens.pop(0)
            return tok, newpos
        return super().next_token(s, pos)

    def _build_operator_free(self, string, tokens):
        # 1) Handle pending tokens first
        if len(tokens) == 1 and tokens[0].special == _SPECIAL_AGENT_RAW:
            raw_token = tokens[0]
            return self.factory.build_constant(raw_token.string, location=raw_token.location)
        return super()._build_operator_free(string, tokens)

    def _token_at(self, s, pos):
        # 1) Match @< @=< @>= @>, keep original behavior
        nxt2 = s[pos:pos+2]
        nxt3 = s[pos:pos+3]
        if nxt2 in ("@<", "@>") or nxt3 in ("@=<", "@>="):
            return super()._token_at(s, pos)

        # 2) Scan single '@' followed raw (whatever content), 
        raw_start = pos + 1
        self.agent_parser.reset(source=s)
        raw_token, raw_end = self.agent_parser.parse(start_pos=raw_start)

        if raw_token is None:
            raise UnexpectedCharacter(s, pos)

        # 3) Inject the raw as a pending token after '@'
        self._pending_tokens.append((raw_token, raw_end))

        # 4) Return the '@' token
        return (
            Token(
                "@", pos,
                atom=False,
                binop=(900, "xfx", self.factory.build_binop),
                unop=(900, "fx",  self.factory.build_unop),
            ),
            raw_start,
        )



# This should be in the parser actually...
class SingleAgentSimpleParser:
    def __init__(self, 
                 case_sensitive=False,
                 source:Optional[str]=None,
    ):
        # Use a integrated context object to manage all states
        self.ctx = Ctx(
            src=source,
            pos=0,
            mark=0,
            state=ParserState(),
            case_sensitive=case_sensitive
        )
        self._recognizers: List[Recognizer] = []  # registered recognizers
        self._active: Optional[Recognizer] = None  # current activated recognizer
        self._out:Optional[Token] = None

    def reset(self, source: Optional[str] = None) -> None:
        """Reset the parser with new context and output dict."""
        self.ctx.reset(src=source)
        self._active = None
        self._out = None

    def add_recognizer(self, recognizer: Recognizer) -> None:
        """Add a recognizer to the parser."""
        if not isinstance(recognizer, Recognizer):
            raise ValueError("Recognizer must be an instance of Recognizer class.")
        self._recognizers.append(recognizer)

    def _emit_pred(self, start: int, end_inclusive: int, raw_meta:str | None = None):
        """Recognizer callback to emit a predicate segment.
        - Flush plain code before the predicate if needed.
        - Emit PRED for [start..end_inclusive].
        - Move mark to end+1 and clear current active recognizer."""
        # emit predicate segment
        self._out = Token(
            self.ctx.ch(start, end_inclusive + 1),
            start,
            atom=True,
            special=_SPECIAL_AGENT_RAW,
        )
        if self._active:
            self._active._is_active = False
            self._active = None

    def parse(self, start_pos:int) -> Tuple[Token,int]:
        """
        Please notice that we start right after the '@' symbol, so the content is expected to be like:
        agent( ... ) or there some space / newlines before agent(...), nothing else.
        """
        self.ctx.reset(pos=start_pos)

        while (not self.ctx.at_end()) and (self._out is None):
            # handle quotes (not escaped)
            self.ctx.handle_quotes()
            self.ctx.handle_comments() 

            # ---------- freeze current context for predicate recognition ----------
            snapshot_pos = self.ctx.pos

            # ---------- active recognizer stepping or try activation ----------
            if self._active:  # inside a predicate
                # step the active recognizer
                consumed = self._active.step(self.ctx, self._emit_pred)
                if consumed:  # skip 
                    continue
                if self.ctx.state.is_in_pure_code:
                    if self.ctx.ch(self.ctx.pos, self.ctx.pos + 2) == ":-":
                        # special case: head-body divider
                        raise ValueError(f"Head-body divider ':-' inside predicate is invalid.")
                    if self.ctx.ch(self.ctx.pos) == ".":
                        # special case: predicate end
                        raise ValueError(f"Predicate end '.' inside predicate is invalid.")
            else:
                for recog in self._recognizers:
                    if recog.try_activate(self.ctx):
                        self._active = recog
                        break
                if (not self._active) and self.ctx.state.is_in_pure_code:
                    # Not activated, and in pure code: if there is any non-space / newline, error out
                    ch = self.ctx.ch(self.ctx.pos)
                    if ch not in (' ', '\t', '\r', '\n'):
                        raise ValueError(f"Agent Name in {_AGENT_NAMES} expected after '@', got '{ch}'.")

            # move forward
            if self.ctx.pos == snapshot_pos:
                self.ctx.fwd(1)

        # ---------- finalization ----------
        # check unclosed quotes/muliti-line comments
        self.ctx.handle_quotes()
        self.ctx.handle_comments()
        if self.ctx.state.in_any_quote:
            raise ValueError(f"Unclosed quotes detected at {self.ctx.ch(self.ctx.pos)}")
        if self.ctx.state.in_multiline_comment:
            raise ValueError(f"Unclosed comments detected at {self.ctx.ch(self.ctx.pos)}")
        
        try:
            out = self._out
            return out, self.ctx.pos + 1 # the last position: use as raw_end
        finally:
            self.reset()
