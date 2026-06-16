"""GBNF grammar for citation-constrained decoding (the M2-deferred feature).

The grammar is generated PER REQUEST from the number of retrieved sources, so a
citation token can only ever be one of [1]..[n_sources] — an out-of-range citation
becomes impossible by construction (vs merely discouraged by the prompt). Prose is
unconstrained except that a bare '[' cannot appear outside a valid citation.
"""
from __future__ import annotations


def build_citation_grammar(n_sources: int) -> str:
    if n_sources <= 0:
        return "root ::= [^\\[]*\n"
    cites = " | ".join(f'"[{i}]"' for i in range(1, n_sources + 1))
    return (
        "root ::= ( text | cite )*\n"
        "text ::= [^\\[]+\n"
        f"cite ::= {cites}\n"
    )
