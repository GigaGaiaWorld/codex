from __future__ import annotations
import re
from typing import Optional, Iterable, Tuple, List
from models.models import (
    Pack,
    PredRaw,
    SpanDraft
)
from models.agent_structure import (
    ToolConfig
)
from abc import ABC, abstractmethod
from dataclasses import replace
from collections import Counter
from models.state import Ctx


# ============================== Predicate Names Initialization ============================== #
def init_names(names:Iterable[str], case_sensitive:bool=False) -> List[str]:
    """
    Initialize and validate the list of predicate/tool names.
    - Remove empty names and strip spaces.
    - Check for illegal characters (only allow letters, digits, underscores, and cannot start with digit).
    - Check for duplicates (case insensitive if case_sensitive=False).
    - Sort predicates from long to short (to prioritize longer matches).
    """
    _ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$") # valid ID pattern
    if isinstance(names, str):
        names = [names]
    if isinstance(names, tuple):
        names = list(names)
    if isinstance(names, set):
        names = list(names)
    if isinstance(names, dict):
        names = list(names.keys())
    # remove space
    raw = [n.strip() for n in names if n and n.strip()]
    if not raw:
        raise ValueError("term cannot be empty")
    # case sensitive?
    norm = raw if case_sensitive else [n.lower() for n in raw]
    # illegal characters in name? 
    bad = [r for r in raw if not _ID_RE.fullmatch(r)]
    if bad:
        raise ValueError(f"Illegal name:{bad}")
    # repeated occurrence?
    dup = [name for name, cnt in Counter(norm).items() if cnt > 1]
    if dup:
        raise ValueError(f"Duplicate terms detected: {sorted(set([raw[i] for i, n in enumerate(norm) if n in dup]))}")
    # sort according to "norm": from long to short
    return [n for _, n in sorted(zip(norm, raw), key=lambda t: len(t[0]), reverse=True)]



# ============================== Recognizers ============================== #
class Recognizer(ABC):
    def __init__(self, 
                 names: Iterable[str] = (),
                 case_sensitive: bool = False
                 ):
        self._is_active:bool = False
        self._current_name: Optional[str] = None
        self.names: List[str] = init_names(
            list(names), 
            case_sensitive=case_sensitive
        )

    # ============================== Utilities ============================== #
    def _is_word_char(self, ch: str) -> bool:
        """word-char for predicate name boundary checking: letters/digits/underscore."""
        return ch != '' and (ch.isalnum() or ch == '_')

    def _match_at(self, ctx: Ctx, pat: str) -> bool:
        """Does the pos of ctx match the pattern (with optional case-sensitivity)."""
        frag = ctx.ch(ctx.pos, ctx.pos + len(pat))
        return frag == pat if ctx.case_sensitive else frag.lower() == pat.lower()

    def reset(self, *args, **kwargs) -> None:
        """Reset the recognizer state."""
        raise NotImplementedError

    # ============================== Abstract Methods ============================== #
    @abstractmethod
    def try_activate(self, ctx: Ctx):
        """If it hits, then self._is_active=True, and record the starting point, etc."""
        raise NotImplementedError
    
    @abstractmethod
    def step(self, ctx: Ctx, emit_pred: callable):
        """Proceed a small step while the function is already enabled.
        - Read/modify ctx.pos (consumer input)
        - To produce PRED: call emit_pred(start, end_inclusive)
        - Returns True if this round has been consumed (the main loop will no longer call _iter())"""
        raise NotImplementedError





# ========================= Recognizer for Predicates ========================== #
# ============================= Predicate Builder ============================== #
class PredBuilder: # 
    __slots__ = ("head","agents")
    def __init__(self) -> None:
        self.head:Pack|None = None
        self.agents:list[Pack] = []

    def set_head(self, name:str, args:str) -> None:
        self.head = Pack(name, args)

    def add_agent(self, name:str, args:str) -> None:
        self.agents.append(Pack(name, args))

    def freeze(self, *, reset = True) -> PredRaw:
        predicate = PredRaw(head=self.head, agents=self.agents[:])
        if reset:
            self.head = None
            self.agents = []
        return predicate
    
# ============================ Two Basic Recognizers ============================ #
# SimpleRecognizer: detect single predicate like langda(...), tool(...), etc.
class SimpleRecognizer(Recognizer):
    """
    Detect: 
        <special_predicate_name>(...) 
    special_predicate_name's are already defined in names:
    tested and works
    """
    def __init__(self, 
                 names:Iterable[str], 
                 *,
                 case_sensitive:bool = False
    ):
        super().__init__(names=names, case_sensitive=case_sensitive)
        self._builder:PredBuilder = PredBuilder()
        self._depth = 0
        self._start = -1
        self._args_start = -1

    def reset(self) -> None:
        self._builder = PredBuilder()
        self._depth = 0
        self._start = -1
        self._args_start = -1

    def try_activate(self, ctx: Ctx) -> bool:

        if not ctx.state.is_in_pure_code:
            return False
        for name in self.names:
            pattern = f'{name}('
            if self._match_at(ctx, pattern):
                prev_char = ctx.ch(ctx.pos - 1)
                if not self._is_word_char(prev_char):  # the previous char is not text or "_"
                    self._is_active = True
                    self._start = ctx.pos
                    self._depth = 1
                    ctx.pos += len(pattern)
                    self._args_start = ctx.pos - 1
                    self._current_name = name
                    return True
        return False

    def step(self, ctx: Ctx, emit_pred: callable) -> bool:
        if not ctx.state.is_in_pure_code:  # comments are allowed
            return False
        if ctx.ch(ctx.pos) == '(':
            self._depth += 1
        elif ctx.ch(ctx.pos) == ')':
            self._depth -= 1
            if self._depth == 0:
                self._builder.add_agent(
                    name = self._current_name, 
                    args = ctx.ch(self._args_start, ctx.pos + 1)
                )
                emit_pred(
                    start = self._start, 
                    end_inclusive = ctx.pos,
                    meta = self._builder.freeze()
                )
                return True
        return False

# ChainRecognizer: detect chain of predicates like weather(... )@agent(... )@tool(... ) => only one agent part allowed currently
class ChainRecognizer(Recognizer):
    """
    Detect: 
        <any_identifier>( ... ) @agent( ... ) [@xxx( ... )] 
    please notice that <any_identifier>( ... ) is just a predicate and should not contain any quote or comment
    from detected '@agent(' pattern, backtrack until the start point 'name(' is found.
    predicates in a row is allowed: @agent(... )@tool(... )@special(... )
    """
    def __init__(
        self,
        names:Iterable[str],
        *,
        case_sensitive:bool = False,
        decorator_mark:str = "@",
    ):
        super().__init__(names=names, case_sensitive=case_sensitive)
        self._decorator_mark = decorator_mark
        self._builder:PredBuilder = PredBuilder()
        
        # define states
        self._chain_start:int = -1
        self._chain_end:int = -1
        self._deco_start:int = -1
        self._deco_depth:int = 0

    def reset(self) -> None:
        self._builder = PredBuilder()
        self._chain_start = -1
        self._chain_end = -1
        self._deco_start = -1
        self._deco_depth = 0

    def _match_any_decorator(self, ctx:Ctx) -> Tuple[Optional[str], int]:
        """Check if current position matches any "decorator + agent" predicate pattern."""
        if ctx.ch(ctx.pos) != self._decorator_mark:
            return None, 0
        
        for pred in self.names:
            pattern = f"{self._decorator_mark}{pred}("
            if self._match_at(ctx, pattern):
                return pred, len(pattern)
        return None, 0

    def _backtrack_predicate_start(self, ctx:Ctx) -> int:
        """Backtrack from decorator "@" to find the start of the agent predicate."""        
        # find the ")" before and ignore spaces
        probe = replace(ctx, pos=ctx.pos - 1)
        head_name = SpanDraft()
        head_args = SpanDraft()
        while probe.pos >= 0 and probe.ch(probe.pos).isspace():
            probe.bwd(1)

        # Found ")", now backtrack to find the matching "("
        if probe.ch(probe.pos) == ")":
            # track the "(...)" structure
            pred_depth = 1
            head_args.close(probe.pos + 1)
            probe.bwd(1)
            while probe.pos >= 0:
                if probe.ch(probe.pos) == ")": 
                    pred_depth += 1
                elif probe.ch(probe.pos) == "(": 
                    pred_depth -= 1 
                    if pred_depth == 0:  # => end of (...) structure
                        head_args.open(probe.pos)
                        head_name.close(probe.pos)
                        probe.bwd(1)
                        # head name part
                        while probe.pos >= 0 and self._is_word_char(probe.ch(probe.pos - 1)):
                            probe.bwd(1)
                        # set head
                        head_name.open(probe.pos)
                        self._builder.set_head(
                            name = ctx.ch(*head_name.dump()), 
                            args = ctx.ch(*head_args.dump())
                        )
                        return probe.pos
                if probe.pos <= 0:
                    break
                probe.bwd(1)
        # No ")" found, try to find simple head without args
        elif self._is_word_char(probe.ch(probe.pos)):
            head_name.close(probe.pos + 1)
            probe.bwd(1)
            while probe.pos >= 0 and self._is_word_char(probe.ch(probe.pos - 1)):
                probe.bwd(1)
            head_name.open(probe.pos)
            self._builder.set_head(
                name = ctx.ch(*head_name.dump()), 
                args = ""
            )
            return probe.pos

        raise ValueError(f"Agent decorator '{self._decorator_mark}'found but no valid predicate head found before it: {ctx.ch(ctx.pos-50, ctx.pos+20)}")
    
    def try_activate(self, ctx:Ctx) -> bool:
        """Try to activate when finding a agent predicate."""
        if not ctx.state.is_in_pure_code:
            return False
        # match decorator pattern
        name, match_len = self._match_any_decorator(ctx)
        if not name:
            return False
        
        predicate_start = self._backtrack_predicate_start(ctx)
        if predicate_start < 0:  # no legal predicate found
            predicate_start = ctx.pos #defensive
        
        # activate and start chain
        self._is_active = True
        self._current_name = name
        self._chain_start = predicate_start
        
        self._deco_depth = 1
        ctx.fwd(match_len)
        self._deco_start = ctx.pos - 1 # "agent(" => start from "("
        self._chain_end = -1
        return True

    def step(self, ctx: Ctx, emit_pred: callable) -> bool:
        """Step through the chain of agent predicates."""
        if not ctx.state.is_in_pure_code: # comments are allowed
            return False
        
        # track depth changes
        if ctx.ch(ctx.pos) == '(':
            self._deco_depth += 1
            return False
        elif ctx.ch(ctx.pos) == ')':
            self._deco_depth -= 1
            if self._deco_depth == 0:
                self._builder.add_agent(
                    name = self._current_name, 
                    args = ctx.ch(self._deco_start, ctx.pos + 1)
                )
                self._chain_end = ctx.pos
            return False
        
        # when depth reaches 0, check for more agent predicates
        if self._deco_depth == 0:
            # handle spaces after current decorator
            while not ctx.at_end() and ctx.ch(ctx.pos).isspace():
                ctx.fwd()
            new_name, new_match_len = self._match_any_decorator(ctx)
            if new_match_len > 0:  # still has other agent predicates
                self._deco_start = ctx.pos + new_match_len - 1 # "agent(" => start from "("
                self._current_name = new_name
                self._deco_depth = 1
                ctx.fwd(new_match_len)
                return False
            
            # no more agent predicates, finish the chain 
            # and store PredLite to "out"
            if self._chain_end >= 0:
                emit_pred(
                    start = self._chain_start, 
                    end_inclusive = self._chain_end, 
                    meta = self._builder.freeze()
                )
            else:  # defensive
                emit_pred(
                    start = self._chain_start, 
                    end_inclusive = ctx.pos - 1, 
                    meta = self._builder.freeze()
                )
            return True
        
        return False




# ========================= Recognizer for Tools ========================== #
class ToolBuilder: # 
    __slots__ = ("classname","content")
    def __init__(self) -> None:
        self.classname:str = None
        self.content:str = None

    def set_tool(self, classname:str, content:str) -> None:
        self.classname = classname
        self.content = content

    def freeze(self, *, reset = True) -> PredRaw:
        tool = ToolConfig(
            classname=self.classname,
            type="Instant",
            content=self.content,
        )
        if reset:
            self.classname = None
            self.content = None
        return tool


# Recognizer for hidden tool calls like __tool_name(...)__
class ToolRecognizer(Recognizer):
    """
    Detect: 
        __<special_tool_name>(...)__
    special_tool_name's are already defined in tool_names:
    tested and works
    """
    def __init__(self, 
                 names:Iterable[str], 
                 *,
                 case_sensitive:bool = False
    ):
        super().__init__(names=names, case_sensitive=case_sensitive)
        self._builder:ToolBuilder = ToolBuilder()
        self._depth = 0
        self._start = -1
        self._args_start = -1

    def reset(self) -> None:
        self._depth = 0
        self._start = -1
        self._args_start = -1

    def try_activate(self, ctx: Ctx) -> bool:

        if not ctx.state.is_in_pure_code:
            return False
        for name in self.names:
            pattern = f'__{name}('
            if self._match_at(ctx, pattern):
                prev_char = ctx.ch(ctx.pos - 1)
                if not self._is_word_char(prev_char):  # the previous char is not text or "_"
                    self._is_active = True
                    self._start = ctx.pos
                    self._depth = 1
                    ctx.pos += len(pattern)
                    self._args_start = ctx.pos
                    self._current_name = name
                    return True
        return False
    
    def step(self, ctx: Ctx, emit_pred: callable) -> bool:
        if not ctx.state.is_in_pure_code:  # comments are allowed
            return False
        if ctx.ch(ctx.pos) == '(':
            self._depth += 1
        elif ctx.ch(ctx.pos) == ')':
            self._depth -= 1
            if self._depth == 0:
                if ctx.ch(ctx.pos + 1, ctx.pos + 3) == '__':
                    self._builder.set_tool(
                        classname = self._current_name, 
                        content = ctx.ch(self._args_start, ctx.pos)
                    )
                    emit_pred(
                        start = self._start, 
                        end_inclusive = ctx.pos + 2,
                        raw_meta = self._builder.freeze()
                    )
                    return True
        return False
    


# ========================= Recognizer for AgentRawParser (for ast parser) ========================== #
class AgentRawRecognizer(Recognizer):
    """
    @agent(<raw_content>)
    Decorator "@" is already detected, now we only need to derect "agent("
    and read until ")"
        
    where raw_content is any content until line break or comment start
    tested and works
    """
    def __init__(self, 
                 names:Iterable[str],
                 *,
                 case_sensitive:bool = False
    ):
        super().__init__(names=names, case_sensitive=case_sensitive)
        self._start = -1
        self._end = -1
        self._depth = 0

    def reset(self) -> None:
        self._start = -1
        self._end = -1
        self._depth = 0

    def try_activate(self, ctx: Ctx) -> bool:
        if not ctx.state.is_in_pure_code:
            return False
        for name in self.names:
            pattern = f'{name}('
            if self._match_at(ctx, pattern):
                self._is_active = True
                self._start = ctx.pos
                self._depth = 1
                ctx.fwd(len(pattern))
                return True
        return False
    
    def step(self, ctx: Ctx, emit_pred: callable) -> bool:
        if not ctx.state.is_in_pure_code:  # comments are allowed
            return False
        if ctx.ch(ctx.pos) == '(':
            self._depth += 1
        elif ctx.ch(ctx.pos) == ')':
            self._depth -= 1
            if self._depth == 0:
                emit_pred(
                    start = self._start, 
                    end_inclusive = ctx.pos,
                    raw_meta = None
                )
                return True
        return False