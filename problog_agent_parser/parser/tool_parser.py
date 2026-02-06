from typing import List, Optional
from .recognizer import Recognizer
from models.state import Ctx, ParserState
from models.models import (
    Extractor,
    ToolTypeAdapter,
    is_valid,
)
from models.tokenizer import ToolArgumentTokenizer
from models.agent_structure import (
    Arguments, ToolConfig,
    IRTDict,
)
"""
1. This module defines parsers for special instant callable terms inside Agent context,
including instant function calls, instant API calls, and lazy/dynamic parameters.
 ===> Hidden calls: __FUNC(Arg1:"...",Arg2:"..."| body => ReturnType)__
The main class is IncontextToolParser, which can parse these instant terms and extract
them into IRTDict as instant tools and instant args.

2. Also, it handles agentic/runtime arguments of langda:
    - Build-time Args: _Var(Var1, Var2)
    - Runtime Args: Api(Var1, Var2=> Return)

we are assuming that the variables are also defined as tools like __Var(Var1, Var2)__
"""

# ==================== Special Hidden Callable Terms Parsers ======================= #
class IncontextToolParser:
    """Base class for instant callable parsers."""
    def __init__(self,
                 extractor:Extractor,
                 case_sensitive:bool=False,
                 ir_dict:Optional[IRTDict]=None,
    ) -> None:

        self.ctx:Ctx = Ctx(
            src=None,
            pos=0,
            mark=0,
            state=ParserState(),
            case_sensitive=case_sensitive
        )
        self.tool_tokenizer = ToolArgumentTokenizer()

        self.extractor = extractor
        self._recognizer: List[Recognizer] = []
        self._active: Optional[Recognizer] = None
        self._agent_term = List[str]
        self._out: IRTDict = ir_dict

    def reset(self, ir_dict: Optional[IRTDict] = None) -> None:
        """Reset the parser with new context and output dict."""
        self.ctx.reset(src=None)
        self._active = None
        self._agent_term = []
        if ir_dict is not None:
            self._out = ir_dict
    
    def add_tool_recognizer(self, recognizer: Recognizer) -> None:
        """Add a recognizer to detect instant tool calls."""
        if not isinstance(recognizer, Recognizer):
            raise ValueError("Recognizer must be an instance of Recognizer class.")
        self._recognizer.append(recognizer)

    def _emit_tool_segment(self, start: int, end_inclusive: int, raw_meta:ToolConfig) -> None:
        """Recognizer callback to emit a tool segment.
        - Flush plain code before the tool if needed.
        - Emit TOOL for [start..end_inclusive].
        - Move mark to end+1 and clear current active recognizer."""
        if self.ctx.mark < start:
            # flush code before tool
            self._agent_term.append(self.ctx.ch(self.ctx.mark, start))

        # Split inner content:
        token_ctx = Ctx(
            src=raw_meta.content,
            case_sensitive=self.ctx.case_sensitive
        )
        self.tool_tokenizer.reset(token_ctx)
        splitter_result = self.tool_tokenizer.tokenize()
        if not splitter_result:
            raise ValueError(f"Failed to tokenize tool inner content: {raw_meta.content}")
        arg_terms, body_str, return_terms = splitter_result
        meta = ToolConfig(
            classname=raw_meta.classname,
            content=raw_meta.content,
            args=arg_terms,
            mode=body_str,
            returns=return_terms
        )

        # emit tool segment
        hash_id, placeholder = self.extractor.render(
            raw_meta.classname + raw_meta.content,
            prefix="args"
        )
        self._agent_term.append(placeholder)
        self._out.add_instant_args(
            name=hash_id,
            body=Arguments(
                hash_id=hash_id,
                conf=meta
            )
        )

        self.ctx.mark = end_inclusive + 1
        if self._active:
            self._active._is_active = False
            self._active = None

    def _process_agent_term(self) -> str:
        """
        Parse instant API calls
        get 'agentic_raws' from IRTDict.all_agentic_raws() method.
        """
        self._agent_term = []
        while not self.ctx.at_end():
            self.ctx.handle_quotes()

            if self.ctx.state.in_any_quote:
                self.ctx.fwd()
                continue

            snapshot_pos = self.ctx.pos
            # Inside an active tool recognition
            if self._active:
                consumed = self._active.step(self.ctx, self._emit_tool_segment)
                if consumed:
                    continue

            else:
                # Try to activate a new tool recognizer
                for recognizer in self._recognizer:
                    if recognizer.try_activate(self.ctx):
                        self._active = recognizer
                        break

            if self.ctx.pos == snapshot_pos:
                self.ctx.fwd()

        self.ctx.handle_quotes()
        # Flush remaining code
        if self.ctx.state.in_any_quote:
            raise ValueError("Unclosed quote detected in source.")
        if self.ctx.mark < self.ctx.pos:
            self._agent_term.append(self.ctx.ch(self.ctx.mark, self.ctx.pos))
        return ''.join(self._agent_term)


    def parse(self) -> IRTDict:
        """
        Main processing function to parse instant tool calls and their variables.
        Returns:
            IRTDict: Updated parameter dict with instant tools and runtime vars.
        """
        # First, process instant tool calls:
        """
        langda(
        Agent:"... __TOOL(Arg1:"...", Arg2:"..." | body => ReturnType) ...",
        ...
        """
        agent_term_str = self._out.builtins.Agent
        self.ctx.reset(src=agent_term_str)
        new_agent_term_str = self._process_agent_term()
        print("Processed Agent Term:", new_agent_term_str)
        # Update the agent term with placeholders
        self._out.builtins_update(
            key="Agent",
            value=new_agent_term_str,
            force=True
        )

        # Second, process agentic/runtime variables:
        """
        _Api(Var1, Var2=> Return),
        _Var(Var1, Var2):"..."
        """
        for hash_id, body in self._out.all_agentic_and_runtime_args().items():
            key = body.raw # original key name => _Api(...) or Api

            if is_valid(ToolTypeAdapter, key):
                class_name, tool_body = ToolTypeAdapter.validate_python(key)
                # Split inner content: args, body, returns
                token_ctx = Ctx(
                    src=tool_body,
                    case_sensitive=self.ctx.case_sensitive
                )
                self.tool_tokenizer.reset(token_ctx)
                splitter_result = self.tool_tokenizer.tokenize()
                if not splitter_result:
                    raise ValueError(f"Failed to tokenize tool inner content: {tool_body}")
                arg_terms, body_str, return_terms = splitter_result
                meta = ToolConfig(
                    classname=class_name,
                    content=tool_body,
                    args=arg_terms,
                    body=body_str,
                    returns=return_terms
                )
            else:
                # Api => regard as parameter only
                meta = ToolConfig(
                    classname="Var",
                    content=""
                )

            # Update tool meta
            body.conf = meta
            self._out.update_args(
                hash_id=hash_id,
                body=body
            )

        try:
            out = self._out
            return out
        finally:
            self.reset()