#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for testing automata."""

# %cd ..

from tools.tools import Tools
from automata.fa.dfa import DFA
from automata.fa.nfa import NFA
from automata.fa.dfa_configuration import DFAConfiguration
from automata.fa.nfa_configuration import NFAConfiguration


# Definire un ASFD che riconosce il linguaggio $L\subseteq\{0,1\}^*$ definito
# come $$L = \{w| w \mbox{ ogni } 0 \mbox{ in } w \mbox{ è seguito
# immediatamente da almeno due } 1 \}$$

dfa = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'0', '1'},
    delta={
        'q0': {'0': 'q0', '1': 'q1'},
        'q1': {'1': 'q2'},
        'q2': {'0': 'q0', '1': 'q2'}
    },
    initial_state='q0',
    final_states={'q0', 'q2'}
)

dfa.draw()

dfa.report_computation(Tools.tokens('001110'))

# Definire un ASFD che riconosce il linguaggio $L\subseteq\{0,1\}^*$ definito
# come $$L = \{w| w\not=\varepsilon \mbox{ e il primo simbolo di } w \mbox{ e
# l'ultimo sono uguali } \}$$

dfa1 = DFA(
    states={'q0', 'q1', 'q2', 'q3', 'q4'},
    input_symbols={'0', '1'},
    delta={
        'q0': {'0': 'q1', '1': 'q2'},
        'q1': {'0': 'q1', '1': 'q3'},
        'q2': {'0': 'q4', '1': 'q2'},
        'q3': {'0': 'q1', '1': 'q3'},
        'q4': {'0': 'q4', '1': 'q1'}
    },
    initial_state='q0',
    final_states={'q1', 'q2'}
)

dfa1.draw()

dfa1.report_computation(Tools.tokens('11'))

# Definire un ASFD che riconosce il linguaggio $L\subseteq\{0,1\}^*$ definito
# come $$L = \{w| \mid w\mid = 7i, i\geq 0 \}$$

dfa2 = DFA(
    states={'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'},
    input_symbols={'0', '1'},
    delta={
        'q0': {'0': 'q1', '1': 'q1'},
        'q1': {'0': 'q2', '1': 'q2'},
        'q2': {'0': 'q3', '1': 'q3'},
        'q3': {'0': 'q4', '1': 'q4'},
        'q4': {'0': 'q5', '1': 'q5'},
        'q5': {'0': 'q6', '1': 'q6'},
        'q6': {'0': 'q0', '1': 'q0'}
    },
    initial_state='q0',
    final_states={'q0'}
)

dfa2.draw()

dfa2.report_computation(Tools.tokens('01010'))

# Definire un ASFD che riconosce il linguaggio $L\subseteq\{0,1\}^*$ definito
# come $$L = \{0,1\}^*- \{\varepsilon\}$$

dfa3 = DFA(
    states={'q0', 'q1'},
    input_symbols={'0', '1'},
    delta={
        'q0': {'0': 'q1', '1': 'q1'},
        'q1': {'0': 'q1', '1': 'q1'}
    },
    initial_state='q0',
    final_states={'q1'}
)

dfa3.draw()

# Non specificare un valore per separator in Tools.tokens() corrisponde ad assumere nessun separatore, e quindi che i simboli siano singoli caratteri

dfa3.report_computation(Tools.tokens('0010'))

# Definire un ASFD che riconosce il linguaggio $L\subseteq\{0,1\}^*$ definito
# come $$L = \{w| w \mbox{ inizia con } 1 \mbox{ e termina con } 0 \}$$

dfa4 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'0', '1'},
    delta={
        'q0': {'1': 'q1'},
        'q1': {'0': 'q2', '1': 'q1'},
        'q2': {'0': 'q2', '1': 'q1'}
    },
    initial_state='q0',
    final_states={'q2'}
)

dfa4.total.draw()

dfa4.report_computation(Tools.tokens('10'))

# Definire un ASFD che riconosce il linguaggio $L\subseteq\{0,1\}^*$ definito
# come $$L = \{w| w \mbox{ contiene un numero pari di } 0, \mbox{ o contiene
# esattamente due } 1 \}$$

dfa5 = DFA(
    states={'q0p', 'q1p', 'q2p', 'q0d', 'q1d', 'q2d'},
    input_symbols={'0', '1'},
    delta={
        'q0p': {'0': 'q0d', '1': 'q1p'},
        'q0d': {'0': 'q0p', '1': 'q1d'},
        'q1p': {'0': 'q1d', '1': 'q2p'},
        'q1d': {'0': 'q1p', '1': 'q2d'},
        'q2p': {'0': 'q2d'},
        'q2d': {'0': 'q2p'}
    },
    initial_state='q0p',
    final_states={'q0p', 'q1p', 'q2p', 'q2d'}
)

dfa5.draw()

dfa5.report_computation(Tools.tokens('101000'))

# Definire un ASFD che riconosce il linguaggio $L\subseteq\{0,1\}^*$ definito
# come $$L = \{w| w \mbox{ contiene esattamente due } 0 \}$$

dfa6 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'0', '1'},
    delta={
        'q0': {'0': 'q1', '1': 'q0'},
        'q1': {'0': 'q2', '1': 'q1'},
        'q2': {'1': 'q2'}
    },
    initial_state='q0',
    final_states={'q2'}
)

dfa6.draw()

dfa6.report_computation(Tools.tokens('101110'))

# Definire un ASFD che riconosce il linguaggio $L\subseteq\{0,1\}^*$ definito
# come
# $$L = \{w| w \mbox{ contiene esattamente due } 0 \mbox{ e almeno due }
# 1\}$$

dfa7 = DFA(
    states={'q00', 'q01', 'q02', 'q10', 'q11', 'q12', 'q20', 'q21', 'q22'},
    input_symbols={'0', '1'},
    delta={
        'q00': {'0': 'q10', '1': 'q01'},
        'q01': {'0': 'q11', '1': 'q02'},
        'q02': {'0': 'q12', '1': 'q02'},
        'q10': {'0': 'q20', '1': 'q11'},
        'q11': {'0': 'q21', '1': 'q12'},
        'q12': {'0': 'q22', '1': 'q12'},
        'q20': {'1': 'q21'},
        'q21': {'1': 'q22'},
        'q22': {'1': 'q22'}
    },
    initial_state='q00',
    final_states={'q22'}
)

dfa7.draw()

dfa7.report_computation(Tools.tokens('10010'))

# Definire un ASFD che riconosce il linguaggio $L\subseteq\{0,1\}^*$ definito
# come $$L = \{w| \mid w\mid mod 5=1\}$$

dfa8 = DFA(
    states={'q0', 'q1', 'q2', 'q3', 'q4'},
    input_symbols={'0', '1'},
    delta={
        'q0': {'0': 'q1', '1': 'q1'},
        'q1': {'0': 'q2', '1': 'q2'},
        'q2': {'0': 'q3', '1': 'q3'},
        'q3': {'0': 'q4', '1': 'q4'},
        'q4': {'0': 'q0', '1': 'q0'},
    },
    initial_state='q0',
    final_states={'q1'}
)

dfa8.draw()

dfa8.report_computation(Tools.tokens('100010'))

# Definire un ASFND avente 3 stati e che riconosce il linguaggio
# $L\subseteq\{0,1\}^*$ definito come $$L = 0^*1^*0^+$$
#
# $L$ è quindi l'insieme delle stringhe composte da una sequenza (eventualmente
# nulla) di 0 seguita da una sequenza (eventualmente nulla) di 1 seguita da una
# sequenza di almeno uno 0.
#

nfa0 = NFA(
    states={'q0', 'q1', 'q2', 'q3'},
    input_symbols={'0', '1'},
    delta={
        'q0': {'0': {'q0'}, '': {'q1'}},
        'q1': {'1': {'q1'}, '': {'q2'}},
        'q2': {'0': {'q3'}},
        'q3': {'0': {'q3'}}
    },
    initial_state='q0',
    final_states={'q3'}
)


nfa0.draw()


nfa0.report_computation(Tools.tokens('00100'))

# Definire un ASFND che riconosce il linguaggio $L\subseteq\{a,b\}^*$ definito
# come $$L_1 = \{ a^nba^m | n,m\geq 0 \}$$

nfa1 = NFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'a', 'b'},
    delta={
        'q0': {'a': {'q0'}, '': {'q1'}},
        'q1': {'b': {'q2'}},
        'q2': {'a': {'q2'}}
    },
    initial_state='q0',
    final_states={'q2'}
)

nfa1.draw()

nfa1.report_computation(Tools.tokens('aabbaa'))

# Definire un ASFND che riconosce il linguaggio $L\subseteq\{a,b\}^*$ definito
# come $$L_1 = \{ a^nba^m | n,m\geq 0 \}$$

nfa2 = nfa1.nfa_no_null

nfa2.draw()

dfa9 = nfa1.dfa

dfa9.draw()
