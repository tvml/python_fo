#!/usr/bin/env python3
"""Classes and methods for working with nondeterministic finite automata."""

import copy
import queue
import random
import graphviz as gv

import tcs.automata.base.automaton_exceptions as ae
import tcs.automata.fa.fa_exceptions as fae
import tcs.automata.fa.fa as fa
import tcs.automata.fa.dfa as dfa
import tcs.automata.fa.nfa_configuration as nfac
import tcs.automata.fa.dfa_in_nfa_configuration as dnfac


class NFA(fa.FA):
    """
    A nondeterministic finite automaton (possibly with epsilon transitions).

    Created by:
        NFA(): definition provided as call parameters
        NFA.load(file): definition provided in yaml file
        NFA.from_dfa(nda): derived from given DFA
        NFA.from_epsilon_nfa(nfa): derived from given NFA by eliminating
        epsilon-transitions
        NFA.from_rg(rg): derived from given regular grammar
        NFA.from_regex(regex): derived from given regular expression
        NFA.union(nfa1, nfa2): union of languages
        NFA.intersection(nfa1, nfa2): intersection of languages
        NFA.difference(nfa1, nfa2): difference of languages
        NFA.concat(nfa1, nfa2): concatenation of languages
        NFA.compl(nfa): complement of language
        NFA.kleene(nfa): Kleene closure of language

    A NFA is coded as follows:
        - states are defined as strings
        - the set of states is a Python set of strings
        - initial state is a string
        - final states is a Python set of strings
        - input symbols are strings
        - transition function is a Python dictionary where
            - keys are strings
            - values are Python dicts where
                - keys are input symbols or the empty string ''
                - values are Python sets of strings
                delta(q1,a)={q2, q3} is coded as delta['q1']['a']={'q2','q3'}
                delta(q1,a)={} is coded as delta['q1']['a']=set()
                delta(q1,epsilon)={q2, q3} is coded as delta['q1']['']={'q2','q3'}
                
    Properties:
        dfa: equivalent dfa
        nfa_no_null: equivalent nfa with no null transition
        complement: nfa accepting complement language
        * kleene: nfa accepting Kleene closure of accepted language
        empty: the accepted language is empty?
        finite: the accepted language is finite?
        universal: the accepted language includes all strings?
        
    Functions:
        * unite: nfa accepting language union of the one accepted by this nfa and the one accepted by given nfa
        intersection: nfa accepting language intersection of the one accepted by this nfa and the one accepted by given nfa
        * concat: nfa accepting language concatenation of the one accepted by this nfa and the one accepted by given nfa
        difference: nfa accepting language difference between the one accepted by this nfa and the one accepted by given nfa
        includes: the given string belongs to the language accepted by this nfa?
        equivalent: is equivalent to given nfa?
        reverse: nfa accepting reverse language
    """

# -----------------------------------------------------------------------------
# Instantiation

    def __init__(self, *, states, input_symbols, delta,
                 initial_state, final_states):
        """Initialize a complete NFA."""
        super().__init__(states, input_symbols, initial_state, final_states)
        self.delta = NFA._transitions_from_delta(delta)
        self.validate()

    @classmethod
    def from_dfa(cls, dfa):
        """Initialize this NFA as one equivalent to the given DFA."""
        return dfa.nfa

    @classmethod
    def from_epsilon_nfa(cls, nfa):
        """
        Initialize from epsilon-NFA.

        Initialize this NFA as one equivalent to the given epsilon-NFA,
        deleting all epsilon transitions.
        """
        return nfa.nfa_no_null

    @classmethod
    def from_regex(cls, re):
        """Initialize this NFA as one equivalent to the given regular expression."""
        return re.nfa

    @classmethod
    def union(cls, nfa1, nfa2):
        """
        Initialize this NFA as one accepting the language union of those accepted
        by the given nfa's.
        """
        return nfa1.unite(nfa2)

    @classmethod
    def intersection(cls, nfa1, nfa2):
        """
        Initialize this NFA as one accepting the language intersection of those accepted
        by the given nfa's.
        """
        return nfa1.intersect(nfa2)
    
    @classmethod
    def difference(cls, nfa1, nfa2):
        """
        Initialize this NFA as one accepting the language difference of those accepted
        by the given nfa's.
        """
        return nfa1.minus(nfa2)

    @classmethod
    def concat(cls, nfa1, nfa2):
        """
        Initialize this NFA as one accepting the language concatenation of those accepted
        by the given nfa's.
        """
        return nfa1.concatenate(nfa2)

    @classmethod
    def compl(cls, nfa):
        """
        Initialize this NFA as one accepting the complement language of the one accepted
        by the given nfa.
        """
        return nfa.complement

    @classmethod
    def kleene(cls, nfa):
        """
        Initialize this NFA as one accepting the kleene-closure of the language accepted
        by the given nfa.
        """
        return nfa.kleene_closure

    @staticmethod
    def _transitions_from_delta(delta):
        """Derive internal representation of transition function."""
        transitions = copy.deepcopy(delta)
        for state, state_transitions in delta.items():
            for input_symbol, is_transitions in state_transitions.items():
                transitions[state][input_symbol] = set()
                for is_transition in is_transitions:
                    transitions[state][input_symbol].add(fa.FATransition(is_transition))
        return transitions

    @property
    def _all_input_symbols(self):
        """Return the set of input symbols plus the epsilon char ''."""
        return self.input_symbols.union({''})

# -----------------------------------------------------------------------------
# Derivation

    @property
    def dfa(self):
        """Return a DFA equivalent to this NFA."""
        # compute equivalent nfa with no epsilon transitions
        nfa = NFA.from_epsilon_nfa(self)
        dfa_initial_state = fa.FA._stringify_states({nfa.initial_state})
        dfa_states = set()
        dfa_symbols = nfa.input_symbols
        dfa_delta = {}
        dfa_final_states = set()
        state_queue = queue.Queue()
        state_queue.put({nfa.initial_state})
        while not state_queue.empty():
            current_nfa_states = state_queue.get()
            current_dfa_state = fa.FA._stringify_states(current_nfa_states)
            if current_dfa_state not in dfa_states:
                # add new dfa state
                dfa_states.add(current_dfa_state)
                # add its entry in delta
                dfa_delta[current_dfa_state] = {}
                # check whether the new state is final in dfa
                if current_nfa_states.intersection(nfa.final_states):
                    dfa_final_states.add(current_dfa_state)
                NFA._next_nfa_states(nfa, current_nfa_states, current_dfa_state,
                                     state_queue, dfa_delta)
        return dfa.DFA(states=dfa_states, input_symbols=dfa_symbols, delta=dfa_delta,
                       initial_state=dfa_initial_state, final_states=dfa_final_states)
        
    @property
    def nfa_no_null(self):
        """Return NFA equivalent to this one, with no null transition."""
        new_nfa = self._complete_delta_missing_values()
        new_delta = new_nfa._delete_epsilon_transitions()
        new_nfa.delta = new_delta
        return new_nfa

    def unite(self, nfa):
        """
        Return a NFA accepting the language union of those accepted
        by this nfa and the one given as parameter.
        """
        # TODO
        nfa1 = None
        return nfa1

    def intersect(self, nfa):
        """
        Return a NFA accepting the language intersection of those accepted
        by this nfa and the one given as parameter.
        """
        return (self.complement.union(nfa.complement)).complement
    
    def minus(self, nfa):
        """
        Return a NFA accepting the language difference of the one accepted
        by this nfa minus the language accepted by the one given as parameter.
        """
        return self.complement.union(nfa).complement

    def concat(self, nfa):
        """
        Return a NFA accepting the language concatenation of those accepted
        by this nfa and the one given as parameter.
        """
        # TODO
        nfa1 = None
        return nfa1

    @property
    def complement(self):
        """
        Return a NFA accepting the complement language of the one accepted
        by this nfa.
        """
        equiv_total_dfa = self.dfa.total
        complement_dfa = copy.deepcopy(equiv_total_dfa)
        complement_dfa.final_states = equiv_total_dfa.states - equiv_total_dfa.final_states
        return dfa.nfa

    @property
    def kleene(self):
        """
        Return a NFA accepting the Kleene-closure of the language accepted by this nfa.
        """
        # TODO
        nfa = None
        return nfa
    
    @property
    def reverse(self):
        """
        Return a NFA accepting the reverse of the language accepted by this nfa.
        """
        return self.dfa.total.reverse

    @classmethod
    def _next_nfa_states(cls, nfa, current_nfa_states, current_dfa_state,
                         state_queue, dfa_delta):
        """Enqueue the next set of current states for the generated DFA."""
        for input_symbol in nfa.input_symbols:
            # consider configuration corresponding to this set of states
            config = nfac.NFAConfiguration.new(current_nfa_states, [input_symbol], nfa)
            # compute new configuration when the input symbol is read
            next_config = nfa._next_configuration(config)
            # derive the next set of states
            next_nfa_states = next_config.states
            if next_nfa_states:
                next_dfa_state = fa.FA._stringify_states(next_nfa_states)
                dfa_delta[current_dfa_state][input_symbol] = next_dfa_state
                state_queue.put(next_nfa_states)

# -----------------------------------------------------------------------------
# Predicates

    @property
    def finite(self):
        """Return True iff the language accepted by this NFA is finite."""
        return self.dfa.finite

    @property
    def empty(self):
        """Return True iff the language accepted by this NFA is empty."""
        return self.dfa.empty
    
    @property
    def universal(self):
        """Return True iff the language accepted by this NFA includes all strings."""
        return self.dfa.universal
    
    def equivalent(self, nfa):
        """Return True if equivalent to the given nfa (same language accepted)."""
        complement_nfa0 = self.complement
        complement_nfa1 = nfa.complement
        intersect_0 = self.intersect(complement_nfa1)
        intersect_1 = nfa.intersect(complement_nfa0)
        return intersect_0.empty and intersect_1.empty

# -----------------------------------------------------------------------------
# Validation

    def _validate_transition_is(self, start_state, input_symbol):
        """
        Check that the input symbol of a transition is valid.

        Verifies that the input symbol of the transition either belongs to the
        input alphabet or is empty (in the case of epsilon transition).
        """
        if input_symbol not in self._all_input_symbols:
            raise fae.InvalidFATransitionError(
                'transition ({},{}) has invalid transition symbol'
                .format(start_state, input_symbol))

    def _validate_transitions_from_state_and_is(self, start_state,
                                                input_symbol,
                                                input_symbol_transitions):
        """
        Check that the transition from a state and symbol is valid.

        Verifies that all resulting states belong to the set of states.
        """
        for transition in input_symbol_transitions:
            new_state = transition.state
            if new_state not in self.states and new_state is not None:
                raise fae.InvalidFATransitionError(
                    'transition ({},{}) has invalid final state {}'
                    .format(start_state, input_symbol, new_state))

# -----------------------------------------------------------------------------
# Completion

    def _complete_transition_missing_symbols(self, state_transitions):
        """Complete transitions from a same state with all symbols."""
        for input_symbol in self.input_symbols:
            if input_symbol not in state_transitions:
                state_transitions[input_symbol] = set()

# -----------------------------------------------------------------------------
# Transformation

    def _delete_epsilon_transitions(self):
        """Delete null transitions from transition function of this nfa."""
        new_delta = {}
        for start_state, transitions in self.delta.items():
            d = {}
            for input_symbol, next_states in transitions.items():
                if input_symbol != '':
                    states = self._state_symbol_epsilon_closure(start_state,
                                                                input_symbol)
                    d[input_symbol] = {fa.FATransition(x) for x in states}
            if d:
                new_delta[start_state] = d
        return new_delta

    def _state_symbol_epsilon_closure(self, start_state, input_symbol):
        """
        Return the set of states reached in a transition in a epsilon-nfa.

        The set of states reached from start_state reading input_symbol
        and possibly following any set of epsilon transitions.
        """
        states = set()
        start_states = self._state_epsilon_closure(start_state)
        for state in start_states:
            transitions = self._transition(state, input_symbol)
            next_states = [x.state for x in transitions]
            for next_state in next_states:
                states = states.union(self._state_epsilon_closure(
                                                    next_state))
        return states

    def _state_epsilon_closure(self, start_state):
        """
        Return the epsilon closure for the given state.

        States are represented as FATransition wrappers.
        The epsilon closure of a state q is the set containing q, along with
        every state that can be reached from q by following only epsilon
        transitions.
        """
        stack = []
        encountered_states = set()
        stack.append(start_state)
        while stack:
            state = stack.pop()
            if state not in encountered_states:
                encountered_states.add(state)
                if '' in self.delta[state].keys():
                    transitions_from_epsilon = self.delta[state]['']
                    next_states_from_epsilon = \
                        [x.state for x in transitions_from_epsilon]
                    stack.extend(next_states_from_epsilon)
        return encountered_states

# -----------------------------------------------------------------------------
# Computation

    def _initial_configuration(self, list_of_tokens):
        """Define the initial configuration of the nfa for the given string."""
        initial_config = nfac.NFAConfiguration.initial_configuration(
                        list_of_tokens=list_of_tokens,
                        automaton=self)
        return self._epsilon_closure(initial_config)

    def _next_configuration(self, current_config):
        """Define the next configuration of the nfa."""
        if not current_config.states:
            raise fae.UndefinedNFATransitionException('no transition defined from empty configuration')
        new_states = set()
        for state in current_config.states_iterator:
            transitions = self._transition(state, current_config.next_token)
            if transitions is not None:
                new_states = new_states.union(
                    {transition.state for transition in transitions})
        next_config = current_config.next_configuration(new_states)
        return self._epsilon_closure(next_config)

    def _epsilon_closure(self, config):
        """
        Return the epsilon closure for the given configuration.

        The epsilon closure of a configuration c is the set containing c,
        along with every configuration that can be reached from c by following
        only epsilon transitions.
        """
        new_states = set()
        n_states = config.number_of_configurations
        if n_states > 0:
            for state in config.states_iterator:
                new_states.add(state)
                try:
                    transitions = self.delta[state]['']
                except KeyError:
                    transitions = set()
            for transition in transitions:
                new_states.add(transition.state)
        config = config.next_epsilon_configuration(new_states)
        if n_states == len(new_states):
            return config
        else:
            return self._epsilon_closure(config)

# -----------------------------------------------------------------------------
# Deterministic paths

    def _deterministic_transition(self, state, input_symbol):
        """
        Follow the transition for the given input symbol on the given state.

        Raise an error if either the state, or the symbol, or the transition
        do not exist.
        """
        if state not in self.states:
            raise ae.InvalidInputError(
                '{} is not a valid state'.format(state))
        if input_symbol not in self.input_symbols and len(input_symbol) > 0:
            raise ae.InvalidInputError(
                '{} is not a valid input symbol'.format(input_symbol))
        try:
            res = self.delta[state][input_symbol]
        except KeyError:
            res = None
        return res

    def _initial_deterministic_configuration(self, list_of_tokens):
        """
        Define the initial configuration of the nfa for the given string.

        The configuration is considered a deterministic one.
        """
        initial_config = dnfac.DFAConfiguration_in_NFA.initial_configuration(
                        list_of_tokens=list_of_tokens,
                        automaton=self)
        return initial_config

    def _next_random_deterministic_configuration(self, current_config):
        """Define the next configuration in a deterministic path of the nfa."""
        new_states = set()
        epsilon_transition_exists = True
        epsilon_transition_applied = True
        next_token = ''
        transitions = self._deterministic_transition(
                current_config.state,
                next_token)
        if transitions is None:
            epsilon_transition_exists = False
        if not epsilon_transition_exists or random.choice([0, 1]) == 0:
            epsilon_transition_applied = False
            next_token = current_config.next_token
            transitions = self._deterministic_transition(
                    current_config.state,
                    next_token)
            if transitions is not None:
                for transition in transitions:
                    new_states.add(transition.state)
        if len(new_states) == 0:
            raise fae.UndefinedNFATransitionException(
                'no transition defined from ({},{})'.format(
                    next_token,
                    current_config.state))
        next_state = random.choice(list(new_states))
        if epsilon_transition_applied:
            next_config = current_config.next_epsilon_configuration(next_state)
        else:
            next_config = current_config.next_configuration(next_state)
        return next_config

# -----------------------------------------------------------------------------
# Other

    def draw(self):
        f = gv.Digraph('nondeterministic_finite_state_machine', engine='dot')
        f.attr(rankdir='LR', size='7,5', fontname='Verdana', style='filled', bgcolor='lightgrey')
        f.node_attr={'color': 'black', 'fillcolor': 'grey', 'style': 'filled'}
        if self.initial_state in self.final_states:
            f.node(self.initial_state, fillcolor='lightblue', shape='doublecircle')
        else:
            f.node(self.initial_state, fillcolor='lightblue', shape='circle')
        f.node_attr['shape']='doublecircle'
        f.attr('node', fillcolor='grey')
        for x in self.final_states:
            f.node(x, shape='doublecircle')
        f.attr('node', shape='circle')
        for source, transitions in self.delta.items():
            dests = {}
            for symbol, destset in transitions.items():
                for dest in destset:
                    if not symbol:
                        symbol = '_'
                    if dest.state in dests.keys():
                        dests[dest.state] = dests[dest.state]+','+symbol
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
            for input_symbol, transitions in sorted(state_transitions.items()):
                if transitions:
                    st = '{'
                    for transition in transitions:
                        st += '{}, '.format(transition)
                    st = st[:-2]
                else:
                    st = '{'
                st += '}'
                s += '\t ({},{}) -> {}\n'.format(start_state, input_symbol, st)
        return s[:-1]
