from typing import List, Optional, Tuple
from models.models import (
    PredRaw
)
from models.state import (
    Ctx,
    ParserState,
    CodeState
)
from .recognizer import Recognizer
from dataclasses import dataclass

# ============================== Segment ============================== #
# Overally used Segment structure
"""
A segment of source code with a specific state.
Metadata can store additional information, such as predicate details.
"""
@dataclass
class Segment:
    span:Tuple[int,int]
    state:CodeState
    meta:Optional[PredRaw] = None  # None or PredLite

    def __repr__(self):
        return f"Segment(span={self.span}, state={self.state})"
    
    def extract(self, text: str) -> str:
        start, end = self.span
        return text[start:end]

    def get_meta(self) -> Optional[list[tuple[str, str]]]:
        if self.meta is None:
            return None
        return self.meta

class UnifiedParser:
    """
    "Segment Parser" for parsing the code into segments with label: code, comment or predicate(agent).
    Main functor: process() => The concrete predicate detection logic should be defined in recognizers.
    """
    def __init__(self, 
                 source: str,
                 *, 
                 case_sensitive=False
    ):
        # Use a integrated context object to manage all states
        self.ctx = Ctx(
            src=source,
            pos=0,
            mark=0,
            state=ParserState(),
            case_sensitive=case_sensitive
        )

        # --- internal states (hidden) ---
        self._recognizers: List[Recognizer] = []  # registered recognizers
        self._active: Optional[Recognizer] = None  # current activated recognizer
        self._out: List[Segment] = []

    # ================================= Tools ================================ #
    def _flush(self, state: CodeState, *, end_inclusive: Optional[int] = None) -> None:
        """Output [mark..end_inclusive] and move the mark to end+1."""
        if end_inclusive is None:
            end_inclusive = self.ctx.pos
        if self.ctx.mark <= end_inclusive:
            self._out.append(Segment((self.ctx.mark, end_inclusive + 1), state))
        self.ctx.mark = end_inclusive + 1
    
    def _emit_pred(self, start: int, end_inclusive: int, meta:PredRaw | None = None):
        """Recognizer callback to emit a predicate segment.
        - Flush plain code before the predicate if needed.
        - Emit PRED for [start..end_inclusive].
        - Move mark to end+1 and clear current active recognizer."""
        if self.ctx.mark < start:
            # flush code before predicate
            self._out.append(Segment((self.ctx.mark, start), self.ctx.code_state()))
        # emit predicate segment
        self._out.append(Segment((start, end_inclusive + 1), CodeState.PRED, meta))
        self.ctx.mark = end_inclusive + 1
        if self._active:
            self._active._is_active = False
            self._active = None

    # ============================== Public APIs ============================== #
    def add_recognizer(self, recognizer: Recognizer) -> None:
        """Add a recognizer to the parser."""
        if not isinstance(recognizer, Recognizer):
            raise ValueError("Recognizer must be an instance of Recognizer class.")
        self._recognizers.append(recognizer)

    def process(self) -> List[Segment]:
        """
            Main parsing process. Returns a list of Segments with metadata.
        """
        quote_start = -1
        comment_start = -1
        
        while not self.ctx.at_end():
            # handle quotes (not escaped)
            quote_change = self.ctx.handle_quotes()
            comment_change = self.ctx.handle_comments() 

            # ---------- handle quote/comment state changes ----------
            if comment_change and self.ctx.state.in_any_comment: # entering comment: flush code before comment
                comment_start = self.ctx.pos
                self._flush(self.ctx.code_state(), end_inclusive=self.ctx.pos - 1)
            elif comment_change and self.ctx.state.is_in_code:
                self._flush(self.ctx.code_state(), end_inclusive=self.ctx.pos)

            elif quote_change and self.ctx.state.in_any_quote:
                quote_start = self.ctx.pos

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
                        raise ValueError(f"Head-body divider ':-' inside predicate is invalid: {self.ctx.ch(quote_start-20,quote_start+20)}")
                    if self.ctx.ch(self.ctx.pos) == ".":
                        # special case: predicate end
                        raise ValueError(f"Predicate end '.' inside predicate is invalid: {self.ctx.ch(quote_start-20,quote_start+205)}")

            else:
                for recog in self._recognizers:
                    if recog.try_activate(self.ctx):
                        self._active = recog
                        break

            # move forward
            if self.ctx.pos == snapshot_pos:
                self.ctx.fwd(1)

        # ---------- finalization ----------
        # check unclosed quotes/muliti-line comments
        self.ctx.handle_quotes()
        self.ctx.handle_comments()
        if self.ctx.state.in_any_quote:
            raise ValueError(f"Unclosed quotes detected at '{self.ctx.ch(quote_start-20,quote_start+20)}'")
        if self.ctx.state.in_multiline_comment:
            raise ValueError(f"Unclosed comments detected starting at '{self.ctx.ch(comment_start-20,comment_start+20)}'")
        # end block
        if self.ctx.mark < self.ctx.length():
            self._flush(self.ctx.code_state())
        return self._out
