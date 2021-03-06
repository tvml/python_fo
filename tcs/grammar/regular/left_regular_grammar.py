#!/usr/bin/env python3
"""Classes and methods for working with right regular grammars."""

import tcs.grammar.base.grammar_exceptions as ge

import tcs.grammar.cf.cf_grammar as cfg
# import tcs.grammar.regular.regular_grammar as rg


class LRG(cfg.CFG):
    """
    A left regular grammar.

    Created by:
        LRG(): definition provided as call parameters
        LRG.load(file): definition provided in json file
        LRG.from_rg(): derived from given (right) regular grammar

    A LRG is coded as follows:
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
                A->a | Aa | '' is coded as productions[('A',)]={('a',), ('A', 'a'), ()}
                
    Properties:
        * rg: equivalent right rg
    """

# -----------------------------------------------------------------------------
# From RG

    @classmethod
    def from_rg(cls, rg):
        """Initialize this RRG as one equivalent to the given (right) regular grammar."""
        return rg.rrg

# -----------------------------------------------------------------------------
# Validation

    def _validate_right_part(self, left_part, right_part):
        """
        Raise an error if the right part of a production is invalid.

        Checks that the right part is a (possibly empty) sequence composed by a non
        terminal followed by a terminal, or by a single terminal
        """
        super()._validate_right_part(left_part, right_part)
        if len(right_part) > 2:
            raise ge.InvalidRightPartError(
                      'right part of production {} -> {} is too long'.
                      format(left_part, right_part))
        if len(right_part) != 0:
            if right_part[-1] not in self.terminals:
                raise ge.InvalidRightPartError(
                     'right part of production {} -> {} is invalid'.
                     format(left_part, right_part))
            if len(right_part) == 2:
                if right_part[0] not in self.non_terminals:
                    raise ge.InvalidRightPartError(
                     'right part of production {} -> {} is invalid'.
                     format(left_part, right_part))

    @property
    def rg(self):
        """Return (right) regular grammar equivalent to this grammar."""
        # TO DO
        rg = None
        return rg