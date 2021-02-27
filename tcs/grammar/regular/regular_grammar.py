#!/usr/bin/env python3
"""Classes and methods for working with right regular grammars."""

import tcs.tools.tools as tools
import tcs.grammar.base.grammar_exceptions as ge

import tcs.grammar.cf.cf_grammar as cfg
import tcs.automata.fa.nfa as nfa


class RG(cfg.CFG):
    """
    A (right) regular grammar.

    Created by:
        RG(): definition provided as call parameters
        RG.load(file): definition provided in json file
        RG.from_lrg(): derived from given left regular grammar
        RG.from_dfa(): derived from given DFA
        RG.from_regex(): derived from given regular expression

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
                
    Properties:
        nfa: equivalent nfa
        * lrg: equivalent left rg
        regex: equivalent regex
        complement: rg generating complement language
        * kleene: rg generating Kleene closure of generated language
        empty: the generated language is empty?
        finite: the generated language is finite?
        universal: the generated language includes all strings?
        
    Functions:
        * unite: rg generating language union of the one generated by this rg and the one generated by given rg
        intersect: rg generating language intersection of the one generated by this rg and the one generated by given rg
        * concat: rg generating language concatenation of the one generated by this rg and the one generated by given rg
        minus: rg generating language difference between the one generated by this rg and the one generated by given rg
        includes: the given string belongs to the language generated by this rg?
        equivalent: is equivalent to given rg?
    """

    @classmethod
    def from_lrg(cls, lrg):
        """Initialize this RG as one equivalent to the given left regular grammar."""
        return lrg.rg

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
        """Return NFA equivalent to this grammar."""
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
    def lrg(self):
        """Return left regular grammar equivalent to this grammar."""
        # TODO
        lrg = None
        return lrg
    
    @property
    def regex(self):
        """Return regex equivalent to this grammar."""
        return self.nfa.dfa.regex
    
    @property
    def complement(self):
        """Return rg generating complement language"""
        return self.nfa.dfa.complement.rg
    
    @property
    def kleene(self):
        """Return rg generating Kleene closure of generated language"""
        return self.nfa.kleene.rg
    
    @property
    def empty(self):
        """Return True iff the generated language is empty"""
        return self.nfa.empty
    
    @property
    def finite(self):
        """Return True iff the generated language is finite"""
        return self.nfa.finite
    
    @property
    def universal(self):
        """Return True iff the generated language includes all strings"""
        return self.nfa.universal

    def unite(self, rg):
        """Return rg generating language union of the one generated by this rg and the one generated by given rg"""
        return (self.nfa.unite(rg.nfa)).rg
    
    def intersect(self, rg):
        """Return rg generating language intersection of the one generated by this rg and the one generated by given rg"""
        return (self.nfa.intersect(rg.nfa)).rg
    
    def concat(self, rg):
        """Return rg generating language concatenation of the one generated by this rg and the one generated by given rg"""
        return (self.nfa.concat(rg.nfa)).rg
    
    def minus(self, rg):
        """Return rg generating language difference between the one generated by this rg and the one generated by given rg"""
        return (self.nfa.minus(rg.nfa)).rg
    
    def includes(self, input):
        """Return True iff the given string belongs to the language generated by this rg"""
        return self.nfa.includes(input)
        
    def equivalent(self, rg):
        """Return True if equivalent to the given rg (same language generated)."""
        return self.nfa.equivalent(rg.nfa)
    
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