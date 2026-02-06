from __future__ import annotations
from enum import Enum
from typing import List, Set, Dict, Union, Any, Optional
from dataclasses import dataclass
from models.models import (
    IncompleteSpanError, 
    SpanDraft, PredRaw, Extractor,
    # validators and adapters
    is_valid, NameTypeAdapter, VarTypeAdapter,
    VarAnoTypeAdapter, QuotedStrTypeAdapter, ListTypeAdapter
)
from models.state import (
    Ctx, ParserState,
)
from models.schemas import (
    _SPECIAL_TERMS
)
from models.tokenizer import (
    ArgumentTokenizer,
)
from models.agent_structure import (
    IRTDict,
    ArgTerm
)
import logging
logger = logging.getLogger(__name__)

"""
Create the initial Intermediate Representation Dictionary (IRTDict)

In this module, the parameters are parsed into key:value pairs by PredicateParser,
but the values are still in string format (not converted to specific types yet).
"""

_ARGS_MARK = "__ARGS__"


# ========================== Special Terms and Settings ========================= #
# ------------------------- Global Parameters and Types ------------------------- #
class PredMode(Enum):
    """Modes of predicate/agent argument parsing."""
    SMPL = "simple" # simple mode: plane positional arguments only
    FULL = "full" # full mode: key:value pair arguments


def is_head_term(term:ArgTerm) -> None:
    """Validate head argument terms."""
    if term.key == _ARGS_MARK:
        return True
    if (
        is_valid(QuotedStrTypeAdapter, term.key) or 
        is_valid(ListTypeAdapter, term.key) or 
        is_valid(NameTypeAdapter, term.key)
    ):
        return True
    else:
        return False

# ============================= Comment Handler ============================= #
class CommentHandler:
    @staticmethod
    def remove_comments(comm_ctx:Ctx) -> str:
        """
        currently in use:
        Remove all comments from the source code in the context.
        Warning: this will modify the source code in the context!
        """
        result = list(comm_ctx.src)
        comment_spandraft:SpanDraft = SpanDraft()
        while not comm_ctx.at_end():
            prev_pos = comm_ctx.pos
            comm_ctx.handle_quotes()
            comment_change = comm_ctx.handle_comments()
            if comment_change and comm_ctx.state.in_any_comment:
                comment_spandraft.open(comm_ctx.pos)
            
            elif comment_change and not comm_ctx.state.in_any_comment:
                comment_spandraft.close(comm_ctx.pos)
                try:
                    start, end = comment_spandraft.dump(reset=True)
                except IncompleteSpanError:
                    raise ValueError(f"Incomplete comment span in source: {comm_ctx.src}")
                # replace comment with spaces
                result[start:end] = [' '] * (end - start)
            if comm_ctx.pos == prev_pos:
                comm_ctx.fwd()

        comm_ctx.reset(src=''.join(result), state=True)

# ============================= Predicate Handling ============================= #
# main Predicate Parser, call with PredicateParser.parse()
class PredicateParser:
    """
    Inside a predicate head(...)@langda(...) or langda(...), we want to extract
    several structured information: key:"value"/[...]/Type pairs
    - raw_meta: original PredRaw object in form:
        PredRaw(
            head=Pack(name:str, args:str),
            agents=List[Pack(name:str, args:str)]
        )

    # Notice:
    head_name and io could be defined both in head(...) and agent(...).
    The priority is: head(...) > agent(...)
    """
    def __init__(self, 
                 extractor:Extractor,
                 case_sensitive:bool,
                 raw_meta:Optional[Union[PredRaw|Any]] = None, 
                 ) -> None:

        # Set prefix for hash ID generation:
        self.prefix: str = "agent"  # Prefix for hash ID generation
        # Initialize the PredicateParser with raw metadata and parsing settings.

        self.raw_meta:PredRaw|None = raw_meta
        self.pack_ctx:Ctx = Ctx(
            src=None,
            pos=0,
            mark=0,
            state=ParserState(),
            case_sensitive=case_sensitive
        )
        # Tools for parsing: comments removal, argument tokenization
        self.comm_handler = CommentHandler()
        self.tokenizer = ArgumentTokenizer()
        # Placeholder extractor: hashing + placeholder generation
        self.extractor = extractor

        # internal states:
        self._mode:PredMode = PredMode.FULL
        self._seen_builtins:Set[str] = set()
        self._out:IRTDict = IRTDict()

    # ============================== Public APIs ============================== #
    def reset(self, raw_meta:Optional[PredRaw]=None) -> None:
        """Reset the parser with new raw_meta."""
        self.raw_meta = raw_meta if raw_meta is not None else self.raw_meta
        self._mode = PredMode.FULL
        self._out = IRTDict()

    def parse(self) -> IRTDict:
        """Main parsing logic."""
        # Generate hash ID and placeholder for the raw_meta
        if self.raw_meta is None:
            raise ValueError("Raw metadata is not set for PredicateParser.")
        self._out.hash_id, self._out.placeholder = self.extractor.render(
            str(self.raw_meta), 
            prefix=self.prefix
        )

        # Parse head if exists
        if self.raw_meta.head:
            self._parse_head()
        else:
            self._out.initials.has_unknown_args = True

        # Parse agents
        for cnt, agent in enumerate(self.raw_meta.agents):
            self._out.initials.agent_name = agent.name

            if cnt >= 1: # Currently only one agent is supported!!!
                raise ValueError("Only one agent is supported per predicate currently...")

            self._parse_agent(agent.args)

        try:
            out = self._out
            return out
        finally:
            self.reset()
    # -------------------------------- Parse Head -------------------------------- #
    def _parse_head(self) -> None:
        """Parse head arguments."""
        head_name = self.raw_meta.head.name
        self._out.initials.head_name = head_name
        self._seen_builtins.add(_SPECIAL_TERMS.NAME.value)

        head_args = self.raw_meta.head.args
        if head_args is None or not head_args.strip():
            return
        # validate basic format: must start with '(' and end with ')'
        if not head_args.startswith('(') or not head_args.endswith(')'):
            raise ValueError(f"Invalid argument format: {self.raw_meta}"
                             f"Expected to start with '(' and end with ')'")
        # check empty content
        if head_args is None or not head_args[1:-1].strip():
            return # empty arguments, just return we have head name only
        self.pack_ctx.reset(src=head_args[1:-1], state=True)
        # Step1: remove comments
        self.comm_handler.remove_comments(self.pack_ctx)

        # Step2: tokenize arguments
        self.tokenizer.reset(self.pack_ctx)
        tokens = self.tokenizer.tokenize()
    
        # Step3: handle head arguments separately
        has_hint = False
        for term in tokens:
            if not is_head_term(term):
                raise ValueError(f"Invalid head argument {term.key} inside: {self.raw_meta}")
            if term.value is not None:
                has_hint = True
            if term.key == _ARGS_MARK:
                self._out.initials.has_unknown_args = True
                if not term.value is None: # not even __ARGS__:"  " allowed
                    raise ValueError(f"Head unknown arguments '{_ARGS_MARK}' cannot have hints: {self.raw_meta}")
                continue # skip this special term
            # validate head argument names:
        self._out.initials.head_args = [term.key for term in tokens if term.key != _ARGS_MARK]

        if has_hint:
            self._seen_builtins.add(_SPECIAL_TERMS.IO.value)
            self._out.builtins_update(
                key='IO',
                value=tokens
            )


    # -------------------------------- Parse Agents -------------------------------- #
    def _parse_agent(self, pack_args:str) -> None:
        """Parse arguments inside a Pack (head or agent)."""
        current_args:Dict = {}

        if not pack_args.startswith('(') or not pack_args.endswith(')'):
            raise ValueError(f"Invalid argument format: {pack_args}"
                             f"Expected to start with '(' and end with ')'")
        if not pack_args[1:-1].strip():
            raise ValueError(f"Empty argument content inside: {pack_args}")
        self.pack_ctx.reset(src=pack_args[1:-1], state=True)

        # Step1: remove comments
        self.comm_handler.remove_comments(self.pack_ctx)

        # Step2: tokenize arguments
        self.tokenizer.reset(self.pack_ctx)
        
        tokens = self.tokenizer.tokenize()

        # Step4: determine parsing mode and parse accordingly
        for count, term in enumerate(tokens):
            if term.key in _SPECIAL_TERMS:
                assert term.key not in self._seen_builtins, f"Built-in term '{term.key}' defined more than once inside: {pack_args}"
                self._seen_builtins.update(term.key)

                # built-in term handling
                if term.key == _SPECIAL_TERMS.NAME.value:
                    """handle Name built-in
                    store in initials.head_name"""
                    if not is_valid(NameTypeAdapter, term.value):
                        raise ValueError(f"Invalid format for 'Name' in '{pack_args}'"
                                         f"Expected a valid NameType.")
                    if self._out.initials.head_name is not None:
                        raise ValueError(f"Head name defined more than once inside: {pack_args}")
                    self._out.initials.head_name = term.value.strip()

                # handle other built-in terms
                elif term.key == _SPECIAL_TERMS.IO.value:
                    """handle IO built-in
                    update in builtins.IO => a list of ArgTerms"""
                    # handle IO built-in
                    if not is_valid(ListTypeAdapter, term.value):
                        raise ValueError(f"Invalid list format for 'IO' in '{pack_args}'"
                                         f"Expected format: [item1, item2, ...]")
                    list_content = term.value.strip()[1:-1].strip()
                    # parse list content:
                    list_ctx = Ctx(
                        src=list_content,
                        case_sensitive=self.pack_ctx.case_sensitive
                    )
                    self.tokenizer.reset(list_ctx)
                    io_terms = self.tokenizer.tokenize()
                    if io_terms:
                        self._out.builtins_update(
                            key='IO',
                            value=io_terms
                        )
                    else:
                        raise ValueError(f"Empty IO list content inside: {pack_args}")
                
                # handle Agent built-in
                elif term.key == _SPECIAL_TERMS.AGENT.value:
                    """Handle Agent built-in
                    update in builtins.Agent => instruction string"""
                    # handle Agent built-in
                    if not is_valid(QuotedStrTypeAdapter, term.value):
                        raise ValueError(f"Invalid 'Agent' instruction format in '{pack_args}'"
                                         f"Expected a quoted string.")
                    agent_instruction = term.value.strip().strip('"').strip("'")
                    self._out.builtins_update(
                        key='Agent',
                        value=agent_instruction
                    )

                # handle Models built-in
                elif term.key == _SPECIAL_TERMS.MODELS.value:
                    # handle Models built-in
                    if not is_valid(ListTypeAdapter, term.value):
                        raise ValueError(f"Invalid list format for 'Models' in '{pack_args}'"
                                         f"Expected format: [item1, item2, ...]")
                    list_content = term.value.strip()[1:-1].strip()
                    # parse list content:
                    list_ctx = Ctx(
                        src=list_content,
                        case_sensitive=self.pack_ctx.case_sensitive
                    )
                    self.tokenizer.reset(list_ctx)
                    model_terms = self.tokenizer.tokenize()

                    self._out.builtins_update(
                        key='Models',
                        value=[term.key for term in model_terms]
                    )


        return
        for count, term in enumerate(tokens):

            # Deal with agent arguments:
            if (count == 0 
                and term.value is None
                and is_head_term(term)
            ):
                    """Logic here: if the first term is a plane argument (string or list), 
                    then it's simple mode. This handle the case:
                    1. head( weather , ... ) => should be simple mode
                    2. agent( [Var1:"hint", _Var(...)], ... ) => should be simple mode
                    3. agent( _Weather, ... ) => should be full mode but starts with plane argument
                    4. agent( Agent:"...", ... ) => should be full mode"""
                    self._mode = PredMode.SMPL

            if self._mode == PredMode.SMPL:
                """Format examples in simple mode:
                  1.langda(weather, [Var1:"hint", _Var(...)], "some instruction").
                  2.agent("some instruction").
                Handle simple mode agent arguments:
                  1. Name Argument: 'weather' => NameTypeAdapter
                  2. Special List Term: '[Var1:"hint", _Var(...)]' => ListTypeAdapter
                  3. Agent Term: '"some instruction"' => StringTypeAdapter"""
                # name argument handling: weather, _llm_, Api123
                if is_valid(NameTypeAdapter, term.key):
                    # handle plane positional arguments
                    head_name = self._out.initials.head_name
                    has_new_head = (head_name is not None and head_name !="_")
                    if has_new_head and head_name != term.key:
                        raise ValueError(
                            f"Head name: '{term.key}' defined more than once inside: "
                            f"{self.raw_meta}\n"
                            f"Or it is a dynamic/lazy parameter in simple mode, which is not allowed."
                            )
                    self._out.initials.head_name = term.key

                # handle special list terms: [Var1:"hint", _Var(...)]
                elif is_valid(ListTypeAdapter, term.key):
                    if current_args.get('Models') is not None:
                        raise ValueError(f"Models or args defined more than once: {pack_args}")
                    list_content = term.key.strip()[1:-1].strip()
                    # parse list content:
                    list_ctx = Ctx(
                        src=list_content,
                        case_sensitive=self.pack_ctx.case_sensitive
                    )
                    self.tokenizer.reset(list_ctx)
                    list_terms = self.tokenizer.tokenize()
                    # models, args_ = ListParser.simple_parse(term.key)

                    if list_terms:
                        for term in list_terms:
                            """!!!!!
                            Be careful here, we haven't planned to
                            use any multi-model setting in simple mode yet.
                            So we only handle args here!!!!
                            """
                            if term.key == 'Models':
                                model = term.value.strip()
                                self._out.builtins_update(
                                    key='Models',
                                    value=model,
                                    force=False
                                )
                            # ======== handle arguments ======== #
                            term_hash_id, _ = self.extractor.render(
                                term.key, 
                                prefix="args"
                            )
                            # Add to IRTDict
                            self._out.add_args(
                                hash_id=term_hash_id,
                                key=term.key,
                                hint=term.value
                            )
                            # ================================== #
                    else:
                        raise ValueError(f"Empty list content inside: {pack_args}")

                # handle agent term: "some instruction"
                elif is_valid(QuotedStrTypeAdapter, term.key):
                    # handle special string terms
                    if current_args.get('Agent') is not None:
                        raise ValueError(f"Agent instruction '{term.key}' defined more than once: {pack_args}")
                    agent_instruction = term.key.strip().strip('"').strip("'")
                    self._out.builtins_update(
                        key='Agent',
                        value=agent_instruction
                    )
                else:
                    raise ValueError(f"Invalid argument {term.key} in simple mode inside: {pack_args}")

            elif self._mode == PredMode.FULL:
                # handle special terms
                if term.key in _SPECIAL_TERMS:
                    """Format examples in full mode:
                      1.langda(Name:"weather", IO:[+Cond:"desc", -Wind:"desc"], Agent:"some instruction", Models:[Model1, Model2]).
                      2.weather(Cond, 11, Temp_c)@langda(Agent:"some instructions", IO:[+Cond:"weather condition as sunny, rainy, cloudy"], Models:[Model1, Model2]).
                    Handle special arguments in full mode: IO, Agent, Models
                      1. Head Name: Name:"weather" => validated as NameType
                      2. IO: [+Cond:"weather good or bad?", +Wind:"wind speed in m/s"] => validated in ListParser => full_parse()
                      3. Agent:"...", instruction string for this agent => validated as string literal
                      4. Models: [Model1, Model2, ...] => validated in ListParser => simple_parse()"""
                    if (term.key == 'IO' or term.key == 'Models') and term.value:
                        # handle special list terms
                        if not is_valid(ListTypeAdapter, term.value):
                            raise ValueError(f"Invalid list format for {term.key}: {term.value}"
                                             f"Expected format: [item1, item2, ...]")
                        list_content = term.value.strip()[1:-1].strip()
                        # parse list content:
                        list_ctx = Ctx(
                            src=list_content,
                            case_sensitive=self.pack_ctx.case_sensitive
                        )
                        self.tokenizer.reset(list_ctx)
                        value = self.tokenizer.tokenize()
                        
                    else:
                        value = term.value.strip().strip('"').strip("'")

                    self._out.builtins_update(
                        key = term.key, 
                        value = value,
                        force=False
                    )

                else:
                    """Handle regular key:value or plane arguments:
                    4. key:value pair argument => 
                        => key validated as VarType or ToolType; value as validated as string literal
                    Store the raw and hint in the IRTDict args.
                    """

                    # ======= handle arguments ======== #
                    hash_id, _ = self.extractor.render(
                        term.value, 
                        prefix="args"
                    )
                    # Add to IRTDict
                    self._out.add_args(
                        hash_id=hash_id,
                        key=term.key,
                        hint=term.value
                    )
                    # ================================== #
            else:
                raise ValueError(f"Unknown parsing mode {self._mode} inside: {pack_args}")