#!/usr/bin/env python3
"""Classes and methods for working with deterministic finite automata."""

import copy
import itertools
import queue
import graphviz as gv

import tools.tools as tools
import automata.fa.fa_exceptions as fae
import automata.fa.fa as fa
import automata.fa.nfa as nfa
import automata.fa.dfa_configuration as dfac
import grammar.regular.regular_grammar as rg


class DFA(fa.FA):
    """
    A deterministic finite automaton.

    Created by:
        DFA(): definition provided as call parameters
        DFA.load(file): definition provided in yaml file
        DFA.from_nfa(nfa): derived from given NFA
        DFA.minimal_dfa(dfa): derived from a dfa by minimization
        *DFA.from_rg(rg): derived from given regular grammar
        *DFA.from_regex(regex): derived from given regular expression

    A DFA is coded as follows:
        - states are defined as strings
        - the set of states is a Python set of strings
        - initial state is a string
        - final states is a Python set of strings
        - input symbols are strings
        - input alphabet is a Python set of strings
        - transition function is a Python dictionary where
            - keys are strings
            - values are Python dicts where
                - keys are input symbols
                - values are strings
                delta(q1,a)=q2 is coded as delta['q1']['a']='q2'
                
    Properties:
        total: equivalent dfa with total transition function
        minimal: minimal equivalent dfa
        nfa: equivalent nfa
        rg: equivalent rg
        * regex: equivalent regex
        empty: the accepted language is empty?
        * finite: the accepted language is finite?
        
    Functions:
        * equivalent: is equivalent to given dfa?

    """

# -----------------------------------------------------------------------------
# Instantiation

    def __init__(self, *, states, input_symbols, delta,
                 initial_state, final_states):
        """Initialize a complete DFA."""
        super().__init__(states, input_symbols, initial_state, final_states)
        self.delta = DFA._transitions_from_delta(delta)
        self.validate()
        self.complete_delta_missing_values()

    @classmethod
    def from_nfa(cls, nfa):
        """Initialize this DFA as one equivalent to the given NFA."""
        return nfa.dfa

    @classmethod
    def from_rg(cls, rg):
        """Initialize this DFA as one equivalent to the given regular grammar."""
        return rg.dfa

    @classmethod
    def from_regex(cls, re):
        """Initialize this DFA as one equivalent to the given regular expression."""
        return re.dfa

    @classmethod
    def minimal_dfa(cls, dfa):
        """Initialize this DFA as a minimal DFA equivalent to the given one."""
        return dfa.minimal

    @classmethod
    def from_dfa_as_total(cls, dfa):
        """
        Initialize this DFA as a DFA with minimum transition function equivalent
        to the given one.
        """
        return dfa.total

    @staticmethod
    def _transitions_from_delta(delta):
        """Derive internal representation of transition function."""
        transitions = copy.deepcopy(delta)
        for state, state_transitions in delta.items():
            for input_symbol, is_transition in state_transitions.items():
                transitions[state][input_symbol] = fa.FATransition(
                    is_transition)
        return transitions

# -----------------------------------------------------------------------------
# Validation

    def _validate_transition_is(self, start_state, input_symbol):
        """
        Check that the input symbol of a transition is valid.

        Verifies that the input symbol of the transition belongs to the input
        alphabet.
        """
        if input_symbol not in self.input_symbols:
            raise fae.InvalidFATransitionError(
                'transition ({},{}) has invalid transition symbol'
                .format(start_state, input_symbol))

    def _validate_transitions_from_state_and_is(self, start_state,
                                                input_symbol,
                                                input_symbol_transition):
        """
        Check that the transition from a state and symbol is valid.

        Verifies that the resulting state belongs to the set of states.
        """
        new_state = input_symbol_transition.state
        if new_state not in self.states and new_state is not None:
            raise fae.InvalidFATransitionError(
                'transition ({},{}) has invalid result state {}'
                .format(start_state, input_symbol, new_state))

# -----------------------------------------------------------------------------
# Completion

    def _complete_transition_missing_symbols(self, state_transitions):
        """Complete transitions from a same state with all symbols."""
        for input_symbol in self.input_symbols:
            if input_symbol not in state_transitions:
                state_transitions[input_symbol] = None

# -----------------------------------------------------------------------------
# Computation

    def _initial_configuration(self, list_of_tokens):
        """
        Return the initial configuration.

        Defines the initial configuration of the dfa for a given input string.
        """
        return dfac.DFAConfiguration.initial_configuration(list_of_tokens=list_of_tokens,
                                                           automaton=self)

    def _next_configuration(self, current_config):
        """
        Return next configuration.

        Defines the next configuration of the dfa for the current configuration
        """
        try:
            transition = self._transition(current_config.state,
                                          current_config.next_token)
        except KeyError:
            raise fae.UndefinedFATransitionException(
                'transition ({},{}) undefined'
                .format(current_config.state,
                        current_config.next_token))
        if transition is None:
            raise fae.UndefinedFATransitionException(
                'transition ({},{}) undefined'
                .format(current_config.state,
                        current_config.next_token))
        else:
            return current_config.next_configuration(transition.state)

# -----------------------------------------------------------------------------
# Derivation

    def _remove_unreachable_states(self):
        """Remove states which are not reachable from the initial state."""
        for state in self.unreachable_states:
            self.states.remove(state)
            self.final_states.remove(state)
            del self.delta[state]

    def _create_markable_pairs_table(self):
        """Create a "markable table" with all combinations of two states."""
        return tools.States_pairs_table(self)

    def _mark_pairs_table_initial(self, table):
        """Mark pairs of states if one is final and one is not."""
        for s in table.keys:
            if any((x in self.final_states for x in s)):
                if any((x not in self.final_states for x in s)):
                    table.set_flag_true(s)

    def _mark_pairs_table_all(self, table):
        """
        Mark additional state pairs.

        A non-marked pair of two states q, q_ will be marked
        if there is an input_symbol a for which the pair
        transition(q, a), transition(q_, a) is marked.
        """
        for pair in table.keys:
            for a in self.input_symbols:
                s0 = self._transition(pair[0], a).state
                s1 = self._transition(pair[1], a).state
                if s0 != s1:
                    new_pair = (s0, s1)
                    if table.flag(new_pair):
                        self._mark_pair(pair, table)
                        break
                    else:
                        if not new_pair == pair:
                            table.add_pair(new_pair, pair)

    def _mark_pair(self, s, table):
        table.set_flag_true(s)
        for s1 in table.pairs(s):
            self._mark_state_table(s1, table)
        table.pairs(s)

    def _join_unmarked_pairs(self, table):
        """Join all overlapping non-marked pairs of states to a new state."""
        # derive the collection of sets of equivalent states
        equivalence_classes = table.unmarked_pairs
        changed = True
        while changed:
            changed = False
            # loop over all couples of sets of equivalent states
            for eq1, eq2 in itertools.combinations(equivalence_classes, 2):
                # check whether the two sets contain a same state
                if not eq2.isdisjoint(eq1):
                    # yes: merge them
                    new_eq = eq1.union(eq2)
                    # remove the old sets from the collection
                    equivalence_classes.remove(eq1)
                    equivalence_classes.remove(eq2)
                    # add the new one
                    equivalence_classes.add(new_eq)
                    # set the changed flag
                    changed = True
                    break
        # finally adjust the DFA
        for eq in equivalence_classes:
            stringified = fa.FA._stringify_states(eq)
            # add the new state
            self.states.add(stringified)
            # copy the transitions from one of the states
            self.delta[stringified] = self.delta[tuple(eq)[0]]
            # replace all occurrences of the old states
            for state in eq:
                self.states.remove(state)
                del self.delta[state]
                for src_state, transition in self.delta.items():
                    for symbol in transition.keys():
                        if transition[symbol].state == state:
                            transition[symbol] = fa.FATransition(stringified)
                if state in self.final_states:
                    self.final_states.add(stringified)
                    self.final_states.remove(state)
                if state == self.initial_state:
                    self.initial_state = stringified

    @property
    def minimal(self):
        """Return a minimal DFA equivalent to this DFA."""
        minimal_dfa = self.total
        minimal_dfa._remove_unreachable_states()
        state_pairs_table = minimal_dfa._create_markable_pairs_table()
        minimal_dfa._mark_pairs_table_initial(state_pairs_table)
        minimal_dfa._mark_pairs_table_all(state_pairs_table)
        minimal_dfa._join_unmarked_pairs(state_pairs_table)
        return minimal_dfa
    
    @property
    def nfa(self):
        """Return NFA equivalent to this DFA."""
        dfa = self
        nfa_delta = {}
        for start_state, transitions in dfa.delta.items():
            nfa_delta[start_state] = {}
            for input_symbol, end_state in transitions.items():
                nfa_delta[start_state][input_symbol] = {end_state}
        return nfa.NFA(states=dfa.states, input_symbols=dfa.input_symbols,
                       delta=nfa_delta, initial_state=dfa.initial_state,
                       final_states=dfa.final_states)

    @property
    def rg(self):
        """Return RG equivalent to this DFA."""
        dfa = self
        terminals = dfa.input_symbols
        nonterminals = {'A'+state for state in dfa.states}
        axiom = 'A'+dfa.initial_state
        productions = {}
        for state, transitions in dfa.delta.items():
            productions[('A'+state,)] = set()
            for symbol, transition in transitions.items():
                productions[('A'+state,)].add((symbol, 'A'+transition.state))
                if state in dfa.final_states:
                    productions[('A'+state,)].add(symbol)
        return rg.RG(terminals=terminals,
                     non_terminals=nonterminals,
                     axiom=axiom,
                     productions=productions)
        
    @property
    def regex(self):
        """Return RegEx equivalent to this DFA."""
        re = None
        # TO DO
        return re

    @property
    def total(self):
        """Return DFA with total transition function equivalent to this one."""
        new_dfa = self.complete_delta_missing_values()
        flag = False
        end_state = fa.FATransition('q*')
        for start_state, transitions in new_dfa.delta.items():
            for k, val in transitions.items():
                if val is None:
                    new_dfa.delta[start_state][k] = end_state
                    flag = True
        if flag:
            new_dfa.delta['q*'] = {k: end_state for k in new_dfa.input_symbols}
            new_dfa.states.add('q*')
        return new_dfa
    
    @property
    def empty(self):
        """Return True iff the language accepted by this DFA is empty."""
        return self.final_states.issubset(self.unreachable_states)
    
    @property
    def finite(self):
        """Return True iff the language accepted by this DFA is finite."""
        # TO Do
        return None

    @property
    def reachable_states(self):
        """Return the states which are reachable from the initial state."""
        rs = set()
        states_to_check = queue.Queue()
        states_checked = set()
        states_to_check.put(self.initial_state)
        while not states_to_check.empty():
            state = states_to_check.get()
            rs.add(state)
            for symbol, dst_state in self.delta[state].items():
                if (dst_state.state not in states_checked) and \
                        (dst_state is not None):
                    states_to_check.put(dst_state.state)
            states_checked.add(state)
        return rs

    @property
    def unreachable_states(self):
        """Return the states which are not reachable from the initial state."""
        return self.states - self.reachable_states

    def equivalent(self, dfa):
        """Return True if equivalent to the given dfa (same language accepted)."""
        # TO DO
        return None
        
# -----------------------------------------------------------------------------
# Other

    def draw(self):
        f = gv.Digraph('finite_state_machine', engine='dot')
        f.attr(rankdir='LR', size='7,5', fontname='Verdana',
               style='filled', bgcolor='lightgrey')
        f.node_attr = {'color': 'black',
                       'fillcolor': 'grey', 'style': 'filled'}
        if self.initial_state in self.final_states:
            f.node(self.initial_state, fillcolor='lightblue',
                   shape='doublecircle')
        else:
            f.node(self.initial_state, fillcolor='lightblue', shape='circle')
        f.node_attr['shape'] = 'doublecircle'
        f.attr('node', fillcolor='grey')
        for x in self.final_states:
            f.node(x, shape='doublecircle')
        f.attr('node', shape='circle')
        for source, transitions in self.delta.items():
            dests = {}
            for symbol, dest in transitions.items():
                if dest.state in dests.keys():
                    dests[dest.state] = dests[dest.state] + ',' + symbol
                else:
                    dests[dest.state] = symbol
            for state, symbols in dests.items():
                f.edge(source, state, symbols)
        return f

    def __str__(self):
        """Return a string representation of the object."""
        s = 'symbols: {}\n'.format(','.join(sorted(self.input_symbols)))
        s += 'states: {}\n'.format(','.join(sorted(self.states)))
        s += 'start state: {}\n'.format(self.initial_state)
        s += 'final states: {}\n'.format(','.join(sorted(self.final_states)))
        s += 'transitions\n'
        for start_state, state_transitions in sorted(self.delta.items()):
            for input_symbol, transition in sorted(state_transitions.items()):
                s += '\t ({},{}) -> {}\n'\
                    .format(start_state, input_symbol, transition)
        return s[:-1]
