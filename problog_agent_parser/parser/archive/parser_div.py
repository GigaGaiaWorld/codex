from typing import List, Optional, Iterable, Tuple
from .models import Segment, CodeState
from .state_track import ParserState, SymbolType
from .predicate import init_predicates
from abc import abstractmethod

class BaseParser:
    """
    "Segment Parser" for parsing the code into ((start, end), type) pairs.
    Main functor: processor()
    The concrete predicate detection logic should be defined in "_boundary_check"
    """
    def __init__(self, 
                 source: str,
                 *, 
                 case_sensitive = False
                 ):
        self.src: str = source

        # --- internal states (hidden) ---
        self._pos:int = 0
        self._mark:int = 0
        self._state:ParserState = ParserState()
        self._in_predicate:str|None = None
        self.case_sensitive:bool = case_sensitive

    # ============================== Utility ============================== #
    def _ch(self, st: int, end: int | None = None) -> str:
        """Return one or multiple characters within bounds."""
        if end is None:
            return self.src[st] if 0 <= st < self._len_src else ''
        st, end = max(0, st), min(self._len_src, end)
        return self.src[st:end] if st < end else ''

    @staticmethod
    def _is_word_char(ch: str) -> bool:
        return ch != '' and (ch.isalnum() or ch == '_')
    
    def _iter(self, i: int = 1) -> None:
        """Move cursor forward by i."""
        self._pos += max(1, i)

    def _match_at_(self, pat:str) -> bool:
        """does the pos match the pattern"""
        if not self.case_sensitive:
            return self._ch(self._pos, self._pos + len(pat)).lower() == pat.lower()
        return self._ch(self._pos, self._pos + len(pat)) == pat
    
    def _flush(self, out: List[Segment], state: CodeState, behind: bool = False) -> None:
        """
        Store the current segment and update markers.
        """
        if behind:
            end = max(0, self._pos - 1)
        else:
            end = self._pos
            self._pos += 1
        if self._mark <= end:
            out.append(Segment((self._mark, end + 1), state))
        self._mark = self._pos

    @property
    def _len_src(self) -> int:
        return len(self.src)

    def _code_state(self) -> CodeState:
        """Private method: decide current code state."""
        if self._state.in_any_comment:
            return CodeState.COMT
        # elif self._in_predicate is not None:
        #     return CodeState.PRED
        return CodeState.CODE

    # ============================== Main Logic ============================== #
    def processor(self) -> List[Segment]:
        """Main parsing process."""
        out:List[Segment] = []

        while self._pos < self._len_src:
            # handle quotes (not escaped)
            if self._ch(self._pos) == '"':
                self._state.switch_state(SymbolType.DOUBLE_QUOTE)
            elif self._ch(self._pos) == "'":
                self._state.switch_state(SymbolType.SINGLE_QUOTE)

            # handle comments
            self._handle_comments(out)

            # handle predicate detection
            self._boundary_check(out)

            # move forward
            self._iter()

        if self._mark < self._len_src:
            out.append(Segment((self._mark, self._len_src), self._code_state()))
        return out

    # ============================== Comment Handling ============================== #
    def _handle_comments(self, out) -> None:
        """Parse single-line and multi-line comments."""
        if self._in_predicate is not None: # Comments are allowed inside predicates 
            return

        if self._ch(self._pos) == '%': # single line comments
            if not self._pos == 0:
                self._flush(out, self._code_state(), behind=True)
            self._state.switch_state(SymbolType.SINGLE_LINE_COMMENT)
        if self._ch(self._pos) == "\n" and self._state.in_single_line_comment: # new line => single line comments end
            self._flush(out, self._code_state())
            self._state.switch_state(SymbolType.NEWLINE)

        if self._ch(self._pos, self._pos + 2) == '/*': # multi-line comments start
            if not self._pos == 0:
                self._flush(out, self._code_state(), behind=True)
            self._state.switch_state(SymbolType.MULTI_LINE_COMMENT_START)
        if self._ch(self._pos, self._pos + 2) == '*/': # multi-line comments end
            self._iter()
            self._flush(out, CodeState.COMT)
            self._state.switch_state(SymbolType.MULTI_LINE_COMMENT_END)

    # ============================== Predicate Handling ============================== #
    @abstractmethod
    def _boundary_check(self, out:List[Segment]) -> None:
        pass



class SimpleParser(BaseParser):
    """
    Detect: 
        <special_predicate_name>(...) 
    special_predicate_name's are already defined in predicate_names
    """
    def __init__(self, 
                 source, 
                 predicate_names = None, 
                 *, 
                 case_sensitive = False
    ):
        super().__init__(source, case_sensitive=case_sensitive)

        # init_predicates returns a List[str] 
        # (already deduplicated and sorted in descending order of length).
        self.predicate_names:List[str] = init_predicates(
            predicate_names or ("langda",),
            case_sensitive=case_sensitive
            )
        self._paren_depth:int = 0

    def _match_predicate(self) -> Tuple[str|None, int]:
        if not self._state.is_in_pure_code:
            return None, 0
        for name in self.predicate_names:
            pattern = f'{name}('
            if self._match_at_(pattern):
                prev = self._ch(self._pos - 1)
                if not self._is_word_char(prev):
                    return name, len(pattern)
        return None, 0

    def _boundary_check(self, out) -> None:

        if self._in_predicate is None:
            name, len_pat = self._match_predicate()
            if name:
                self._flush(out, self._code_state(), behind=True)
                self._in_predicate = name
                self._paren_depth = 1
                self._iter(len_pat - 1)
                return

        if self._in_predicate is not None and self._state.is_in_pure_code:
            if self._ch(self._pos) == '(':
                self._paren_depth += 1
            elif self._ch(self._pos) == ')':
                self._paren_depth -= 1
                if self._paren_depth == 0:
                    self._flush(out, CodeState.PRED)
                    self._in_predicate = None
                    return



class ChainParser(BaseParser):
    """
    Detect: 
        <any_identifier>( ... ) @agent( ... ) [@xxx( ... )] 
    please notice that <any_identifier>( ... ) is just a predicate and should not contain any quote or comment
    from detected '@agent(' pattern，backtract until the start point 'name(' is found.
    decorators in a row is allowed：@agent(... )@tool(... )@special(... )
    """
    def __init__(
        self,
        source:str,
        *,
        case_sensitive:bool = False,
        decorators:Iterable[str] = ("agent",),   # expandable: ("agent","tools","...")
        decorator_mark:str = "@",
    ):
        super().__init__(source, case_sensitive = case_sensitive)

        self._decorators = set(d if case_sensitive else d for d in decorators)
        self._decorator_mark = decorator_mark
        # define states
        self._chain_start:int = -1
        self._chain_end:int = -1
        self._deco_depth:int = 0

    def _match_any_decorator(self) -> Tuple[str|None, int]:
        if not self._ch(self._pos) == self._decorator_mark:
            return None, 0
        for deco in sorted(self._decorators, key=len, reverse=True):
            pattern = f"{self._decorator_mark}{deco}("
            if self._match_at_(pattern):
                prev = self._ch(self._pos-1)
                if self._is_word_char(prev): # prevent case: langda(...) <=> pattern: gda
                    continue
                return deco, len(pattern)
        return None, 0

    def _backtrack_predicate_start(self) -> int:
        bt = self._pos - 1 # back tracker: self._pos == decorator "@"
        # find the ")" before and ignore spaces
        while bt >= 0 and self._ch(bt).isspace():
            bt -= 1
        if not self._ch(bt) == ")":
            return -1
        # track the "(...)" structure
        depth = 0
        while bt >= 0:
            if self._ch(bt) == ")": 
                depth += 1
            elif self._ch(bt) == "(": 
                depth -= 1 
                if depth == 0: # => end of (...) structure
                    bt -= 1 # back track predicate name
                    while self._is_word_char(self._ch(bt)):
                        bt -= 1
                    return bt + 1
            bt -= 1
        return -1

    def _start_chain(self, out: List[Segment], name:str, match_len: int, predicate_start: int) -> None:
        if self._mark < predicate_start:
            # (mark, temp_pos)|(predicate_start, ..., predicate_end)
            temp_pos = self._pos
            self._pos = predicate_start
            self._flush(out, self._code_state(), behind=True)
            # (predicate_start|(mark, pos), ..., predicate_end)
            self._pos = temp_pos
            self._mark = predicate_start

        self._in_predicate = name
        self._chain_start = predicate_start
        self._deco_depth = 1
        self._iter(match_len - 1)
        self._chain_end = -1

    def _finish_chain(self, out: List[Segment], end_pos_inclusive: int) -> None:
        if self._chain_start < 0 or end_pos_inclusive < self._chain_start:
            # reset by failure
            self._in_predicate = None
            self._deco_depth = 0
            self._chain_start = -1
            self._chain_end = -1
            return
        # (predicate_start|(mark, ..., pos)|predicate_end)
        temp_pos = self._pos
        self._pos = end_pos_inclusive + 1
        self._flush(out, CodeState.PRED, behind=True)
        self._pos = temp_pos
        self._in_predicate = None
        self._deco_depth = 0
        self._chain_start = -1
        self._chain_end = -1
        
    def _boundary_check(self, out:List[Segment]):
        if not self._state.is_in_pure_code:
            return

        if self._in_predicate is None: # detect new predicate
            name, match_len = self._match_any_decorator()
            if name:
                predicate_start = self._backtrack_predicate_start()
                if predicate_start < 0: # no legal predicate found
                    return
                self._start_chain(out, name, match_len, predicate_start)
            return
        
        # intern depth changes...
        if self._ch(self._pos) == '(':
            self._deco_depth += 1
            return
        elif self._ch(self._pos) == ')':
            self._deco_depth -= 1
            if self._deco_depth == 0:
                self._chain_end = self._pos
            return
        
        if self._deco_depth == 0: 
            new_name, new_match_len = self._match_any_decorator() 
            if new_match_len > 0: # still has other terms?
                self._in_predicate = new_name
                self._deco_depth += 1
                self._iter(new_match_len - 1)
                return

            if self._chain_end >= 0:
                self._finish_chain(out, self._chain_end)
            else: # defensive!!!
                self._finish_chain(out, self._pos - 1)