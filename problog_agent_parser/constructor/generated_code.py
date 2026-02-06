from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from problog.logic import Term, Clause

@dataclass
class Candidate:
    """
    A structure to hold the transit head and candidate terms for a specific agent.
    - transit_head: The intermediate term used in the probabilistic logic.
    - candidate_terms: A list of clauses representing different candidate definitions.
    """
    transit_head: Term
    candidate_terms: List[Clause] = field(default_factory=list)


@dataclass
class CandidateDicts:
    """
    A structure to hold candidate terms for probabilistic agents.
    Each candidate is associated with a unique hash_id.
    """
    candidates: Dict[str, Candidate] = field(default_factory=dict)


# 不要死磕ast了, 我们使用1=>1, 直接用dict来表示就行了
candidates = {
    "hash_id": {
        "transit_head": Term(
            "fest_agent_head_hash_id",
            Term(...), # 和实际的real_head参数项对应
            location=None,
            **{}
        ),
        "candidate_terms": [
            Clause(
                Term(
                    "real_head",
                    Term(...), # candidate term 1
                    location=None,
                    **{}
                ),
                [
                    Term("candidate_head", Term(...), location=None, **{}),
                    Term("=", Term(...), Term("rainy"), location=None, **{})
                ],
                location=None,
                **{}
            ),
            Clause(
                Term(
                    "real_head",
                    Term(...), # candidate term 2
                    location=None,
                    **{}
                ),
                [
                    Term("0.4::candidate_head", Term(...), location=None, **{}),
                ],
                location=None,
                **{}
            ),
            Clause(
                Term(
                    "other_head",
                    Term(...), # candidate term 3
                    location=None,
                    **{}
                ),
                [
                    Term("0.5::candidate_head", Term(...), location=None, **{}),
                ],
                location=None,
                **{}
            ),
        ]
    }
}



"""
weather("sunny", Arg2) :-
    fest_agent_head_hash_id1(Arg1, Arg2), 
    ...



# End of Code

fest_agent_head_hash_id1(Arg1, Arg2) :-
    candidate_head(Arg1), Arg2 = "rainy".

0.4::candidate_head(Arg1) :-
    weather("sunny", Arg1).

0.5::candidate_head(M) :-
    weather("cloudy", M).
"""

