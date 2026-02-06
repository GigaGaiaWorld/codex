from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

# ============================== Parser State ============================== #
# The lexical state of the parser, tracking quotes and comments.
# ========================= #================== #=========================== #
class CodeState(Enum):
    CODE = "CODE" # in code segment, not in predicate segment
    COMT = "COMT" # in comment segment
    PRED = "PRED" # in code segment, and also in predicate segment

class SymbolType(Enum):
    SINGLE_LINE_COMMENT = "SCMT"
    NEWLINE = "NEWLINE"
    MULTI_LINE_COMMENT_START = "MCMT_START"
    MULTI_LINE_COMMENT_END = "MCMT_END"
    SINGLE_QUOTE = "SINGLE_QUOTE"
    DOUBLE_QUOTE = "DOUBLE_QUOTE"

@dataclass(slots=True)
class ParserState:
    in_single_line_comment: bool = False
    in_multiline_comment: bool = False
    in_single_quote: bool = False
    in_double_quote: bool = False

    def reset_states(self) -> None:
        """Reset all parser states to default (not in any comment or quote)."""
        self.in_single_line_comment = False
        self.in_multiline_comment = False
        self.in_single_quote = False
        self.in_double_quote = False

    def switch_state(self, update:SymbolType) -> 'ParserState':
        """Update the parser state based on the current symbol.
        Args:
            update (SymbolType): The type of symbol encountered.
        Returns:
            ParserState: The updated parser state.
        """
        # Single line comment:
        if update == SymbolType.SINGLE_LINE_COMMENT:
            if not self.in_any_quote and not self.in_multiline_comment:
                self.in_single_line_comment = True
        elif update == SymbolType.NEWLINE:
            self.in_single_line_comment = False

        # Multiple line comment:
        elif update == SymbolType.MULTI_LINE_COMMENT_START:
            if not self.in_any_quote and not self.in_single_line_comment:
                self.in_multiline_comment = True
        elif update == SymbolType.MULTI_LINE_COMMENT_END:
            if self.in_multiline_comment:
                self.in_multiline_comment = False
        
        # Quotes:
        elif update == SymbolType.SINGLE_QUOTE:
            if not self.in_any_comment and not self.in_double_quote:
                self.in_single_quote = not self.in_single_quote
        elif update == SymbolType.DOUBLE_QUOTE:
            if not self.in_any_comment and not self.in_single_quote:
                self.in_double_quote = not self.in_double_quote

        return self
    
    @property
    def in_any_quote(self) -> bool:
        return self.in_single_quote or self.in_double_quote

    @property
    def in_any_comment(self) -> bool:
        return self.in_single_line_comment or self.in_multiline_comment

    @property
    def is_in_code(self) -> bool:
        """Determine if the current state is within code (not in comment)."""
        return not self.in_any_comment

    @property
    def is_in_pure_code(self) -> bool:
        return not self.in_any_comment and not self.in_any_quote


# ============================== Unified Context ============================== #
# The unified context owns the source buffer and cursor, and manages the parser state.
# =========================== #================== #============================ #
@dataclass(slots=True)
class Ctx:
    """Unified parsing context that owns the source buffer and cursor.
    - src: source code buffer
    - pos: current scanning cursor
    - mark: boundary for the next emitted Segment
    - state: lexical state (quotes/comments)
    - case_sensitive: matching mode for recognizers (fallback default)"""
    src:str
    pos:int = 0
    mark:int = 0
    state:ParserState = field(default_factory=ParserState)
    case_sensitive: bool = False

    # ============================== Public Tools ============================== #
    def reset(self, *, src:str=None, state:bool=True, pos:int=0) -> None:
        """
        Reset position, mark and state snapshot in one shot (used by parser before recognizers).
        Args:
            src (str, optional): New source buffer. If None, keep the current one.
            state (bool, optional): Whether to reset the parser state. Defaults to True.
        """
        if src is not None:
            self.src = src
        self.pos = pos
        self.mark = pos
        if state:
            self.state.reset_states() # reset comments/quotes

    def ch(self, st:int, end:Optional[int] = None) -> str:
        """Return one or multiple characters within bounds."""
        n = len(self.src)
        if end is None:
            return self.src[st] if 0 <= st < n else ''
        st, end = max(0, st), min(n, end)
        return self.src[st:end] if st < end else ''

    def length(self) -> int:
        """Total length of source buffer."""
        return len(self.src)

    def fwd(self, i:int = 1) -> None:
        """Move cursor forward by i (at least 1)."""
        self.pos += max(1, i)

    def bwd(self, i:int = 1) -> None:
        """Move cursor backward by i (at least 1)."""
        self.pos += min(-1, -i)

    def at_end(self) -> bool:
        """
        Check if cursor reaches end of buffer.
        """
        return self.pos >= self.length()

    # ============================== Public Handlers ============================== #
    def code_state(self) -> CodeState:
        """decide current code state."""
        if self.state.in_any_comment: 
            return CodeState.COMT
        return CodeState.CODE
    
    def handle_quotes(self) -> bool:
        """Parse quotes (not escaped).
        - Returns True if a quote state is switched.
        """
        if self.state.in_any_comment:
            return False
        old_state = self.state.in_any_quote
        if self.ch(self.pos) == '"':
            self.state.switch_state(SymbolType.DOUBLE_QUOTE)
            
        elif self.ch(self.pos) == "'":
            self.state.switch_state(SymbolType.SINGLE_QUOTE)

        if old_state != self.state.in_any_quote:
            return True
        return False

    def handle_comments(self) -> bool:
        """Parse single-line and multi-line comments.
        - active: current active recognizer, only use for allowing comments inside predicates.
        - Returns True if a comment state is switched.
        """
        # single line comments
        old_state = self.state.in_any_comment
        if self.ch(self.pos) == '%':
            self.state.switch_state(SymbolType.SINGLE_LINE_COMMENT)
        # new line => single line comments end
        if self.ch(self.pos) == "\n" and self.state.in_single_line_comment:
            self.state.switch_state(SymbolType.NEWLINE)

        # multi-line comments start
        if self.ch(self.pos, self.pos + 2) == '/*':
            self.state.switch_state(SymbolType.MULTI_LINE_COMMENT_START)
        # multi-line comments end
        if self.ch(self.pos, self.pos + 2) == '*/':
            self.state.switch_state(SymbolType.MULTI_LINE_COMMENT_END)
            self.fwd(2)
            
        if old_state != self.state.in_any_comment:
            return True
        return False

 