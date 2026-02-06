from typing import List, Optional, Iterable
from .models import Segment, CodeState
from .state_track import ParserState, SymbolType
from .predicate import init_predicates

class Parser:
    """
    Parse the code into ((start, end), type) pairs.
    Main function: processor()
    The parser detects comments, quotes, and user-defined predicates (e.g., langda).
    """
    def __init__(self, 
                 source: str, 
                 predicate_names: Optional[Iterable[str]] = None,
                 *,
                 case_sensitive:bool = False,
                 ):
        self.src: str = source
        # init_predicates returns a List[str] 
        # (already deduplicated and sorted in descending order of length).
        self.predicate_names: List[str] = init_predicates(
            predicate_names or ("langda",),
            case_sensitive=case_sensitive
            )

        # --- internal states (hidden) ---
        self._pos: int = 0
        self._mark: int = 0
        self._state: ParserState = ParserState()
        self._in_predicate_name: str|None = None
        self._paren_depth: int = 0

    # ============================== Utility ============================== #
    def _ch(self, st: int, end: int | None = None) -> str:
        """Return one or multiple characters within bounds."""
        if end is None:
            return self.src[st] if 0 <= st < self._len_src else ''
        st, end = max(0, st), min(self._len_src, end)
        return self.src[st:end] if st < end else ''

    def _iter(self, i: int = 1) -> None:
        """Move cursor forward by i."""
        self._pos += max(1, i)

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
        elif self._in_predicate_name is not None:
            return CodeState.PRED
        else:
            return CodeState.CODE

    # ============================== Main Logic ============================== #
    def processor(self) -> List[Segment]:
        """Main parsing process."""
        out: List[Segment] = []

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
        if self._ch(self._pos) == '%':
            if not self._pos == 0:
                self._flush(out, self._code_state(), behind=True)
            self._state.switch_state(SymbolType.SINGLE_LINE_COMMENT)

        if self._ch(self._pos, self._pos + 2) == '/*':
            if not self._pos == 0:
                self._flush(out, self._code_state(), behind=True)
            self._state.switch_state(SymbolType.MULTI_LINE_COMMENT_START)

        if self._ch(self._pos, self._pos + 2) == '*/':
            self._iter()
            self._flush(out, CodeState.COMT)
            self._state.switch_state(SymbolType.MULTI_LINE_COMMENT_END)

        if self._ch(self._pos) == "\n" and self._state.in_single_line_comment:
            self._flush(out, self._code_state())
            self._state.switch_state(SymbolType.NEWLINE)

    # ============================== Predicate Handling ============================== #
    def _is_word_char(self, ch: str) -> bool:
        return ch != '' and (ch.isalnum() or ch == '_')

    def _match_predicate(self):
        if not self._state.is_in_pure_code:
            return None, 0
        for name in self.predicate_names:
            pattern = f'{name}('
            if self._ch(self._pos, self._pos+len(pattern)) == pattern:
                prev = self._ch(self._pos - 1)
                if not self._is_word_char(prev):
                    return name, len(pattern)
        return None, 0

    def _boundary_check(self, out) -> None:

        if self._in_predicate_name is None:
            name, len_pat = self._match_predicate()
            if name:
                self._flush(out, self._code_state(), behind=True)
                self._in_predicate_name = name
                self._paren_depth = 1
                self._iter(len_pat - 1)
                return

        if self._in_predicate_name is not None and self._state.is_in_pure_code:
            if self._ch(self._pos) == '(':
                self._paren_depth += 1
            elif self._ch(self._pos) == ')':
                self._paren_depth -= 1
                if self._paren_depth == 0:
                    self._flush(out, CodeState.PRED)
                    self._in_predicate_name = None
                    return
