from typing import List, Tuple, Optional
from enum import Enum
from models.state import Ctx
from models.schemas import (
    _ALLOWED_BRS,
    _ALLOWED_DPS,
)
from models.models import (
    is_valid,
    VarTypeAdapter
)
from models.agent_structure import (
    ArgTerm
)
import warnings


_SPLITERS = {
    "body": "|",
    "return": "=>"
}


# ============================= Depth Tracker ============================= #
class DepthTracker:
    """
    Track the depth of nested brackets while parsing arguments.
    """
    _BR_PAIRS = {"(":")", "[":"]", "{":"}"}

    def __init__(self) -> None:
        self._br_stack:List[str] = []
    
    @property
    def _depth(self) -> int:
        """Get current depth of brackets."""
        return len(self._br_stack)
    
    def reset(self) -> None:
        """Reset the bracket stack."""
        self._br_stack = []

    def push(self, ctx:Ctx) -> None:
        """Push an opening bracket onto the stack."""
        br = ctx.ch(ctx.pos)
        if br not in _ALLOWED_BRS.keys():
            raise ValueError(f"Unsupported opening bracket {br} inside: {ctx.src}")
        if self._depth > _ALLOWED_DPS:
            raise ValueError(f"Exceeded maximum allowed bracket depth inside: {ctx.src}")
        self._br_stack.append(br)

    def pop(self, ctx:Ctx) -> None:
        """Pop the last opening bracket from the stack and check for matching."""
        br = ctx.ch(ctx.pos)
        if not self._br_stack:
            raise ValueError(f"Unmatched closing bracket {br} inside: {ctx.src}")
        if not self._BR_PAIRS[self._br_stack[-1]] == br:
            raise ValueError(
                f"Mismatched closing bracket {br} inside: {ctx.src}"
                f"expected {self._BR_PAIRS[self._br_stack[-1]]} but got {br}"
                )
        self._br_stack.pop()
    

# ============================= Argument Tokenizer ============================= #
class ArgumentTokenizer:
    """
    Tokenize arguments inside head/agent, considering quotes and brackets.
    E.g., (arg1, key1:value1, key2:[val2a, val2b], key3:func(arg1, arg2))
    """
    def __init__(self) -> None:
        self.depth_tracker = DepthTracker()
        self._colon:int = -1
        self._tokens:List[ArgTerm] = []
        self.ctx:Ctx|None = None

    # def set_context(self, ctx:Ctx) -> None:
    #     """
    #     Set the parsing context.
    #     Must be done before calling tokenizer().
    #     """
    #     self.ctx = ctx

    def reset(self, ctx:Optional[Ctx]=None) -> None:
        """Reset the tokenizer state."""
        self.ctx = ctx
        # self.depth_tracker.reset()
        # self._colon = -1
        # self._tokens = []

    def _emit_term(self) -> ArgTerm:
        """Emit the current term as ArgTerm."""
        term_str = self.ctx.ch(self.ctx.mark, self.ctx.pos).strip()
        if not term_str:
            raise ValueError(f"Empty argument detected in agent arguments inside: {self.ctx.src}")

        if self._colon < self.ctx.mark:
            key = term_str.strip()
            val = None
        else:
            # key:value pair term
            key = self.ctx.ch(self.ctx.mark, self._colon).strip()
            val = self.ctx.ch(self._colon + 1, self.ctx.pos).strip()
        self.ctx.mark = self.ctx.pos + 1  # skip ','
        self._colon = -1 # reset colon position
        # Create ArgTerm:
        term = ArgTerm(
            key=key,
            value=val,
            transit=is_valid(VarTypeAdapter, val)
        )
        self._tokens.append(term)

    def tokenize(self) -> List[ArgTerm]:
        """
        tokenize the arguments inside head/agent.
        has to be in ( ... ) format.
        Returns:
            List[ArgTerm]: List of tokenized argument terms.
        """

        self.depth_tracker.reset()
        self._colon = -1
        self._tokens = []

        while not self.ctx.at_end(): # until ')'
            ch = self.ctx.ch(self.ctx.pos)
            if ch.isspace():
                self.ctx.fwd()
                continue

            # handle quotes and comments: comments are already removed
            _ = self.ctx.handle_quotes()
            # If in any quote, skip...
            if self.ctx.state.in_any_quote:
                self.ctx.fwd()
                continue

            # divide by top-level commas:
            if ch in _ALLOWED_BRS.keys():
                self.depth_tracker.push(self.ctx)
            elif ch in _ALLOWED_BRS.values():
                self.depth_tracker.pop(self.ctx)
            
            if self.depth_tracker._depth == 0 and not self.ctx.state.in_any_quote:
                if ch == ':': # colon shows up
                    if self._colon >= self.ctx.mark:
                        raise ValueError(f"Multiple colons detected in agent arguments inside: {self.ctx.src}")
                    self._colon = self.ctx.pos
                if ch == ',': # 
                    self._emit_term()

            self.ctx.fwd()

        if self.depth_tracker._depth != 0:
            # Unmatched opening brackets, for example, opening with '(' but no closing bracket
            raise ValueError(f"Unmatched opening brackets in agent arguments inside: {self.ctx.src}")
        if not self.ctx.state.is_in_pure_code:
            raise ValueError(f"Unclosed quotes or comments in agent arguments inside: {self.ctx.src}")

        # emit the last term
        if self.ctx.pos > self.ctx.mark:
            self._emit_term()

        try:
            out = self._tokens
            return out
        finally:
            self.reset()

class CurrentTerm(Enum):
    ARGS = "args"
    BODY = "body"
    RETURN = "return"

# =================== Tool Inner Content Splitter ======================= #
class ToolArgumentTokenizer:
    def __init__(self):
        self.splitter_ctx:Ctx|None = None
        self.tokenizer = ArgumentTokenizer()
        self.current_term:CurrentTerm = CurrentTerm.ARGS

    def reset(self, ctx:Optional[Ctx]=None) -> None:
        self.splitter_ctx = ctx

    def tokenize(self) -> Tuple[List[ArgTerm], str, List[str]]:
        """
        Split the inner content of a callable into args, body, and return type.
        for a tool like:
        def Api(arg1:str, arg2:int, arg3:dict): ... return output1, output2
        Default set could be: (Arg1:"content",Arg2:23 | body => Output1)
        Allowed forms:
            args
            args : body => outputs
            args : body
            args        => outputs
                 : body => outputs
                 : body 

        Final return:
            args_str, body_str, rtype_str
        """
        arg_str:str = ""
        body_str:str = ""
        return_str:str = ""
        self.current_term = CurrentTerm.ARGS

        while not self.splitter_ctx.at_end():
            self.splitter_ctx.handle_quotes()

            if self.splitter_ctx.state.in_any_quote:
                self.splitter_ctx.fwd()
                continue
            
            # Check spliters: args | body => Type
            if self.splitter_ctx.ch(self.splitter_ctx.pos, self.splitter_ctx.pos + len(_SPLITERS["body"])) == _SPLITERS["body"]:
                if self.current_term == CurrentTerm.BODY or self.current_term == CurrentTerm.RETURN:
                    raise ValueError(f"Inproper format: body already defined before body/return spliter at {self.splitter_ctx.src}")
                self.current_term = CurrentTerm.BODY
                arg_str = self.splitter_ctx.ch(self.splitter_ctx.mark, self.splitter_ctx.pos).strip()
                self.splitter_ctx.fwd(len(_SPLITERS["body"]))
                self.splitter_ctx.mark = self.splitter_ctx.pos
                continue

            if self.splitter_ctx.ch(self.splitter_ctx.pos, self.splitter_ctx.pos + len(_SPLITERS["return"])) == _SPLITERS["return"]:
                if self.current_term == CurrentTerm.RETURN:
                    raise ValueError(f"Inproper format: return already defined before return spliter at {self.splitter_ctx.src}")
                if self.current_term == CurrentTerm.ARGS:
                    arg_str = self.splitter_ctx.ch(self.splitter_ctx.mark, self.splitter_ctx.pos).strip()
                elif self.current_term == CurrentTerm.BODY:
                    body_str = self.splitter_ctx.ch(self.splitter_ctx.mark, self.splitter_ctx.pos).strip()
                self.current_term = CurrentTerm.RETURN
                self.splitter_ctx.fwd(len(_SPLITERS["return"]))
                self.splitter_ctx.mark = self.splitter_ctx.pos
                continue

            self.splitter_ctx.fwd()

        # Final term:
        if self.current_term == CurrentTerm.ARGS:
            arg_str = self.splitter_ctx.ch(self.splitter_ctx.mark, self.splitter_ctx.pos).strip()
        elif self.current_term == CurrentTerm.BODY:
            body_str = self.splitter_ctx.ch(self.splitter_ctx.mark, self.splitter_ctx.pos).strip()
        elif self.current_term == CurrentTerm.RETURN:
            return_str = self.splitter_ctx.ch(self.splitter_ctx.mark, self.splitter_ctx.pos).strip()

        # Tokenize args: it need to be in ( ... ) format, so we add parentheses here
        self.splitter_ctx.reset(src=arg_str)
        self.tokenizer.reset(self.splitter_ctx)

        try:
            arg_terms = self.tokenizer.tokenize()
            for i, term in enumerate(arg_terms):
                arg_terms[i] = ArgTerm(
                    key=term.key,
                    value=term.value.strip("'").strip('"') if term.value else None,
                    transit=term.transit
                )
        except:
            warnings.warn(f"Failed to tokenize arguments in hidden tool: {self.splitter_ctx.src}. Assuming not a tool.")
            return None
        
        # Tokenize returns:
        return_terms = return_str.split(",") if return_str else []
        return_terms = [ret.strip() for ret in return_terms if ret.strip()]

        try:
            out = (list(arg_terms), body_str, list(return_terms))  # 或 tuple(...)
            return out
        finally:
            self.reset()



