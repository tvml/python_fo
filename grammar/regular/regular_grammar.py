#!/usr/bin/env python3
"""Classes and methods for working with left regular grammars."""

import tools.tools as tools
import grammar.base.grammar_exceptions as ge

import grammar.cf.cf_grammar as cfg
# import grammar.regular.right_regular_grammar as rrg
import automata.fa.nfa as nfa
# import regexpr.regex_exceptions as rex


class RG(cfg.CFG):
    """
    A (left) regular grammar.

    Created by:
        RG(): definition provided as call parameters
        RG.load(file): definition provided in json file
        *RG.from_rrg(): derived from given right regular grammar
        *RG.from_dfa(): derived from given DFA
        *RG.from_regex(): derived from given regular expression

    A RG is coded as follows:
        - terminals are defined as strings
        - the set of terminals is a Python set of strings
        - nonterminals are defined as strings
        - the set of nonterminals is a Python set of strings
        - axiom is a string
        - productions is a Python dictionary where
            - keys are Python tuples of strings of length one
            - values are Python dicts where
                - keys are input symbols, including the empty string ''
                - values are sets of possibly empty Python tuples of strings
                A->a | aA | '' is coded as productions[('A',)]={('a',), ('a', 'A'), ()}
    """

    @classmethod
    def from_rrg(cls, right_rg):
        """Initialize this RG as one equivalent to the given right regular grammar."""
        return right_rg.rg

    @classmethod
    def from_dfa(cls, dfa):
        """Initialize this RG as one equivalent to the given DFA."""
        return dfa.rg

    @classmethod
    def from_regex(cls, regex):
        """Initialize this RG as one equivalent to the given regular expression."""
        return regex.rg


# -----------------------------------------------------------------------------
# Derivation

    @property
    def nfa(self):
        """Return DFA equivalent to this grammar."""
        grammar = self
        states = {'q'+nt for nt in grammar.non_terminals}
        final_state = 'q'+tools.Tools.new_nt_symbol(grammar.non_terminals)
        states.add(final_state)
        initial_state = 'q'+grammar.axiom
        final_states = {final_state}
        if grammar.null_string_produced:
            final_states.add(initial_state)
        delta = {}
        for state in states:
            delta[state] = {}
        for left_part,right_parts in grammar.productions.items():
            for right_part in right_parts:
                start_state = 'q'+left_part[0]
                if len(right_part) == 0:
                    pass
                elif len(right_part) == 1:
                    if right_part[0] in delta[start_state].keys():
                        delta[start_state][right_part[0]].add(final_state)
                    else:
                        delta[start_state][right_part[0]] = {final_state}
                else:
                    if right_part[0] in delta[start_state].keys():
                        delta[start_state][right_part[0]].add('q'+right_part[1])
                    else:
                        delta[start_state][right_part[0]] = {'q'+right_part[1]}
        return nfa.NFA(states=states,
                    input_symbols = grammar.terminals,
                    delta=delta,
                    initial_state = initial_state,
                    final_states=final_states)

    @property
    def rrg(self):
        """Return right regular grammar equivalent to this grammar."""
        # TO DO
        rrg = None
        return rrg

# -----------------------------------------------------------------------------
# Validation

    def _validate_right_part(self, left_part, right_part):
        """
        Raise an error if the right part of a production is invalid.

        Checks that the right part is a (possibly empty) sequence composed by a
        terminal followed by a non terminal, or by a single terminal
        """
        super()._validate_right_part(left_part, right_part)
        if len(right_part) > 2:
            raise ge.InvalidRightPartError(
                      'right part of production {} -> {} is too long'.
                      format(left_part, right_part))
        if len(right_part) == 1:
            if right_part[0] not in self.terminals and left_part[0] != "S'":
                raise ge.InvalidRightPartError(
                     'right part of production {} -> {} is invalid'.
                     format(left_part, right_part))
        if len(right_part) == 2:
            if right_part[1] not in self.non_terminals:
                raise ge.InvalidRightPartError(
                     'right part of production {} -> {} is invalid'.
                     format(left_part, right_part))
