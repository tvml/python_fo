#!/usr/bin/env python3
"""Classes and methods for working with regular expressions."""

import random

import tcs.base.base as base
import tcs.tools.tools as tools
import tcs.regexpr.regex_exceptions as re
import tcs.automata.fa.nfa as nf
import tcs.grammar.cf.cf_grammar as cfg
import tcs.grammar.regular.regular_grammar as rg
import tcs.parse.rd.recursive_descent as rdp
import tcs.grammar.cf.syntax_tree as syt
import string
import copy


class RegEx(base.Base):
    """
    A regular expression.

        Created by:
        RegEx(): definition provided as call parameters
        Regex.from_dfa(): from dfa
        Regex.from_rg(): from rg
        
    Properties:
        * nfa: equivalent nfa
        rg: equivalent rg
        rrg: equivalent rrg
        

    """

    def __init__(self, *, alphabet, expression):
        """Initialize a complete Turing machine."""
        self.alphabet = alphabet.copy()
        self.input_expression = expression
        self.expression = self.canonical
        self.grammar = self._set_grammar()
        self.all_chars = tools.Tools.all_chars(self.alphabet)
        self.validate()
        self._syntax_tree = self.syntax_tree
        self.includes_null = self._includes_null()
        
    @classmethod
    def from_dfa(cls, dfa):
        """Initialize this RegEx as one equivalent to the given dfa."""
        return dfa.regex
    
    @classmethod
    def from_rg(cls, rg):
        """Initialize this RegEx as one equivalent to the given RG."""
        return rg.regex

    @property
    def canonical(self):
        """
        Derive a canonical regular expression from the one in input.

        An equivalent regular expression is derived where operators are always
        explicit and that does not assume operator precedence or left
        associativity. The structure of the expression is only based on
        parenthesization.
        """
        terminals = self.alphabet
        tokens = self.input_expression
        if len(tokens) == 0:
            return []
        l1 = []
        # add dots
        for i, v in enumerate(tokens):
            if i > 0:
                if v in terminals.union({'('}) and\
                        tokens[i-1] in terminals.union({'*'}):
                    l1.append('.')
                if v in terminals and tokens[i-1] == ')':
                    l1.append('.')
                if v == '(' and tokens[i-1] in terminals:
                    l1.append('.')
            l1.append(v)
        lst = l1
        l1 = []
        # insert parentheses around +, double parentheses, add initial and
        # final parentheses
        for i, v in enumerate(lst):
            if v == '+':
                l1.extend([')', v, '('])
            elif v in {'(', ')'}:
                l1.extend([v, v])
            else:
                l1.append(v)
        l1.insert(0, '(')
        l1.insert(0, '(')
        l1.append(')')
        l1.append(')')
        lst = l1
        # insert parenthesis to model left associativity
        plus_stack = []
        dot_stack = []
        stack = []
        l1 = []
        for i, v in enumerate(lst):
            if v == '+':
                if len(plus_stack) == 0 or plus_stack[-1] != len(stack):
                    plus_stack.append(len(stack))
                else:
                    l1.insert(stack[-1], '(')
                    l1.append(')')
            elif v == '.':
                if len(dot_stack) == 0 or dot_stack[-1] != len(stack):
                    dot_stack.append(len(stack))
                else:
                    l1.insert(stack[-1], '(')
                    l1.append(')')
            elif v == '(':
                stack.append(len(l1))
            elif v == ')':
                if len(plus_stack) > 0 and plus_stack[-1] == len(stack):
                    plus_stack.pop()
                if len(dot_stack) > 0 and dot_stack[-1] == len(stack):
                    dot_stack.pop()
                stack.pop()
            l1.append(v)
        lst = l1
        # reduce the number of parentheses wherever possible, to
        # simplify expression
        stack = []
        l1 = []
        cp_counter = 0
        for i, v in enumerate(lst):
            if v == '(':
                stack.append(len(l1))
                l1.append('(')
            elif v == ')':
                if l1[-1] in terminals and l1[-2] == '(':
                    # simplify (a) to a
                    l1.pop(stack[-1])
                    stack.pop()
                elif l1[-2] in terminals and l1[-1] == '*' and l1[-3] == '(':
                    # simplify (a*) to a*
                    l1.pop(stack[-1])
                    stack.pop()
                else:
                    # ) closes an expression of more than one terminals
                    # increase counter of the length of ) sequence
                    cp_counter += 1
            else:
                if cp_counter > 0:
                    # this character immediately follows a sequence of )
                    # of length cp_counter
                    k = 1
                    while k < cp_counter and stack[-1] == stack[-2] + k:
                        # delete a ( from the list to be returned and
                        # increase the number of ) not considered
                        l1.pop(stack[-2])
                        stack.pop(-2)
                        k += 1
                    # there are still cp_counter-k+1 ) to insert
                    for j in range(cp_counter-k+1):
                        l1.append(')')
                        stack.pop()
                    cp_counter = 0
                l1.append(v)
        while cp_counter > 0:
            if len(stack) == 1 or stack[-1] > stack[-2] + 1:
                l1.append(')')
            else:
                l1.pop(stack[-1])
            stack.pop()
            cp_counter -= 1
        return l1

    def _set_grammar(self):
        """Return CFG associated to RE definition."""
        grammar = cfg.CFG(
            terminals=self.alphabet.union({'(', ')', '+', '*', '.'}),
            non_terminals={'S', 'A', 'B'},
            axiom='S',
            productions={
                'S': {'A'},
                'A': self.alphabet.union({('(', 'A', ')'),
                                          ('(', 'A', '+', 'A', ')'),
                                          ('(', 'A', '.', 'A', ')'),
                                          ('B', '*')}),
                'B': self.alphabet.union({('(', 'A', '.', 'A', ')'),
                                          ('(', 'A', '+', 'A', ')'),
                                          ('(', 'A', ')')}),
                },
            no_null_production=True,
            null_string_produced=True
            )
        return grammar

    def validate(self):
        """Return True if this RG is internally consistent."""
        if not self._check_syntax():
            raise re.InvalidREStructure('Syntactically incorrect RE')
        return True

    def _check_syntax(self):
        """Return True if this RG is syntactically correct."""
        d = rdp.RD_parser(self.grammar).get_derivations(self.expression)
        if len(d) == 1:
            self._syntax_tree = d[0].syntax_tree
        return len(d) == 1

    @property
    def syntax_tree(self):
        """Return the syntax tree associated to the RE structure."""
        if self._syntax_tree is None:
            self._syntax_tree = rdp.RD_parser(self.grammar).parse(self.expression)
        return self._syntax_tree

    def _includes_null(self):
        """Return true if the null string belongs to the associated language."""
        def check_null(node):
            if isinstance(node, syt.Terminal_node):
                return False
            else:
                if len(node.children) == 1:
                    # S->A
                    return check_null(node.children[0])
                elif len(node.children) == 2:
                    # A->B*
                    return True
                elif len(node.children) == 3:
                    # A-> (A), B->(A)
                    return check_null(node.children[1])
                elif node.children[2].symbol == '+':
                    # A->(A+A), B->(A+A)
                    return check_null(node.children[1]) or\
                        check_null(node.children[3])
                else:
                    # A->(A.A), B->(A.A)
                    return check_null(node.children[1]) and\
                        check_null(node.children[3])

        if not self.expression:
            return True
        else:
            st = self.syntax_tree
            return check_null(st.root)

# -----------------------------------------------------------------------------
# Derivation

    @property
    def nfa(self):
        """Return NFA equivalent to this regular expression."""
        
        re = copy.deepcopy(self)
        
        alphabet = list(string.ascii_lowercase) #lista di tutti i simboli alfabetici minuscoli
        nfa_dict = {} #dizionario degli automi nfa
        
        # Trasformo l'espressione regolare in forma postfissa
        postfix_re = re.postfix()  
        
        # Creo gli nfa che riconoscono i singoli simboli dell'alfabeto dell'espressione regolare e li inserisco nel diz. degli automi
        for t in re.alphabet:
            nfa_dict[t] = nf.NFA.create_nfa_from_symbol(t)
        
        while (len(postfix_re) > 1):
                
            # Cerco la prima lettera dell'alfabeto non ancora utilizzata per creare automi
            i=0
            while (alphabet[i] in nfa_dict.keys()):
                i+=1
            new_alfa = alphabet[i]
            
            # Scorro l'espr. reg. in forma postfissa finchè non incontro un operatore
            j=0
            while (postfix_re[j] in nfa_dict.keys()):
                j+=1        
                
            # Se l'operatore letto è un "+" eseguo l'unione dei due operandi che precedono l'operatore; l'automa ottenuto sarà corrispondente
            # alla lettera new_alfa e gli stati dell'automa vengono rinominati tutti con la lettera new_alfa come iniziale;
            # Le lettere corrispondenti ai due automi di cui ho fatto l'unione e l'operatore "+" vengono eliminati e sostituiti dalla lettera
            # corrispondente all' "automa unione"
            if (postfix_re[j] == '+'):
                nfa_dict[new_alfa] = nfa_dict[postfix_re[j-2]].union(nfa_dict[postfix_re[j-2]],nfa_dict[postfix_re[j-1]])
                nfa_dict[new_alfa] = nf.rename_states(nfa_dict[new_alfa], new_alfa)
                postfix_re[j-2] = new_alfa
                for k in range (j-1, len(postfix_re)-2):
                    postfix_re[k] = postfix_re[k+2]
                postfix_re.pop()
                postfix_re.pop()
    
            
            # Se l'operatore letto è un "." eseguo la concatenazione dei due operandi che precedono l'operatore; l'automa ottenuto sarà corrispondente
            # alla lettera new_alfa e gli stati dell'automa vengono rinominati tutti con la lettera new_alfa come iniziale;
            # Le lettere corrispondenti ai due automi di cui ho fatto la concatenazione e l'operatore "." vengono eliminati e sostituiti dalla lettera
            # corrispondente all' "automa concatenazione"
            elif postfix_re[j] == '.':
                nfa_dict[new_alfa] = nf.NFA.concat(nfa_dict[postfix_re[j-2]],nfa_dict[postfix_re[j-1]])
                nfa_dict[new_alfa] = nf.rename_states(nfa_dict[new_alfa], new_alfa)
                postfix_re[j-2] = new_alfa
                for k in range (j-1, len(postfix_re)-2):
                    postfix_re[k] = postfix_re[k+2]
                postfix_re.pop()
                postfix_re.pop()
            
            # Se l'operatore letto è un "*" eseguo l'iterazione dell'operando che lo precede; l'automa ottenuto sarà corrispondente
            # alla lettera new_alfa e gli stati dell'automa vengono rinominati tutti con la lettera new_alfa come iniziale;
            # La lettera corrispondente all'automa di cui ho fatto l'iterazione e l'operatore "*" vengono eliminati e sostituiti dalla lettera
            # corrispondente all' "automa concatenazione"
            else:
                
                nfa_dict[new_alfa] = nfa_dict[postfix_re[j-1]].kleene(nfa_dict[postfix_re[j-1]])
                nfa_dict[new_alfa] = nf.rename_states(nfa_dict[new_alfa], new_alfa)
                postfix_re[j-1] = new_alfa
                for k in range (j, len(postfix_re)-1):
                    postfix_re[k] = postfix_re[k+1]
                postfix_re.pop()
        return (nfa_dict[postfix_re[0]])


    @property
    def rrg(self):
        """Return Right RG equivalent to this regular expression."""
        return self.rg.rrg

    @property
    def rg(self):
        """Return (left) regular grammar equivalent to this regular expression."""
        return self.nfa.dfa.rg

# -----------------------------------------------------------------------------
# Other

    def correct_string(self, s):
        """Returns true iff the given string is described by the regex"""
        cc, c, a = self.nfa.dfa.compute(s)
        return a

    @property
    def postfix(self):
        """Derive the postfix notation of the regular expression given in input."""
    
        def post_order(node):
            s=[]
            if type(node).__name__=='Terminal_node':
                s.append(node.symbol)
            else:
                if len(node.children) == 5:
                    s+=post_order(node.children[1])
                    s+=post_order(node.children[3])
                    s.append(node.children[2].symbol)
                elif len(node.children) == 3:
                    s+=post_order(node.children[1])
                elif len(node.children) == 2:
                    s+=post_order(node.children[0])
                    s.append(node.children[1].symbol)
                else:
                    s+=post_order(node.children[0])
            return tuple(s)
        
        return post_order(self.syntax_tree.root)
        # operands = self.alphabet #insieme degli simboli (operandi) dell'espressione regolare
        # expression = self.expression 
        # operator_prio = {'+': 0, '.': 1, '*': 2} # dizionario che ha come chiavi gli operatori e come valori le loro priorità
        
        # postfix_str = [] #stringa finale in notazione postfissa
        # operator_stack = [] #pila degli operatori

        # for t in expression:
        
        #     # Condizione 1: se leggo un operando lo inserisco in "postfix_str"
        #     if t in operands:
        #         postfix_str.append(t)
                
        #     # Condizione 2: se la pila degli operatori è vuota o contiene una "(" in cima 
        #     # allora inserisco l'operatore che sto leggendo nella pila
        #     elif ((len(operator_stack) == 0) or (operator_stack[-1] == '(')):    
        #         operator_stack.append(t)
                
        #     # Condizione 3: se leggo una "(" la inserisco sempre nella pila    
        #     elif (t == '('):
        #         operator_stack.append(t) 
            
        #     # Condizione 4: se leggo una ")", estraggo gli operatori dalla pila 
        #     # e li scrivo in "postfix_str" finché non trovo una "(" che elimino senza copiarla
        #     elif (t == ')'):
        #         while operator_stack[-1] != '(':
        #             postfix_str.append(operator_stack[-1])
        #             operator_stack.pop()
        #         operator_stack.pop()
                
        #     # Condizione 5: se la priorità dell'operatore che sto leggendo è 
        #     # maggiore o uguale a quella dell'operatore in cima alla pila, allora lo inserisco nella pila
        #     elif (operator_prio[t] >= operator_prio[operator_stack[-1]]):
        #         operator_stack.append(t) 
        
        #     # Condizione 6: sto leggendo un operatore che ha priorità più bassa rispetto all'operatore in cima alla pila.
        #     # In questo caso estraggo l'elemento in cima alla pila e lo inserisco in "postfix_str".
        #     # Ripeto il confronto con il nuovo elemento in cima alla pila.    
        #     else: 
        #         while len(operator_stack) != 0 and operator_prio[t] < operator_prio[operator_stack[-1]] and operator_stack[-1] in list(operator_prio.keys()):
        #             postfix_str.append(operator_stack[-1])
        #             operator_stack.pop()
        #         operator_stack.append(t)
    
        # # Dopo aver letto tutta l'espressione regolare, se rimangono operatori nella pila 
        # # li estraggo e li inserisco in "postfix_str" secondo la logica "LIFO"
        # while (len(operator_stack) > 0):
        #     postfix_str.append(operator_stack[-1])
        #     operator_stack.pop()                 
    
        # return postfix_str
    
    def equivalent(self, regex):
        """Return true if this regex is equivalent to the one given in input."""
        return self.nfa.equivalent(regex.nfa)

    def random_string(self, iteration_probability=.8,
                      iteration_probability_decrease=.65):
        """Return a random string described by the regular expression."""
        rs = self._random_string_from_subtree(self.syntax_tree.root,
                                              iteration_probability,
                                              iteration_probability_decrease)
        if self.all_chars:
            return ''.join(rs)
        else:
            return tools.Tools.phrase(rs)

    def _random_string_from_subtree(self, current, iteration_probability,
                                    iteration_probability_decrease):
        if isinstance(current, syt.Terminal_node):
            rs = [current.symbol]
        else:
            nt = current.symbol
            if nt == 'S':
                # S->A
                rs = self._random_string_from_subtree(current.children[0],
                                                      iteration_probability,
                                                      iteration_probability_decrease)
            elif nt == 'A':
                if current.children[0].symbol == 'B':
                    # A->B*
                    rs = []
                    check_value = random.random()
                    check_probability = iteration_probability
                    while check_value < check_probability:
                        rs.extend(self._random_string_from_subtree(
                            current.children[0],
                            iteration_probability,
                            iteration_probability_decrease))
                        check_probability *= iteration_probability_decrease
                        check_value = random.random()
                elif len(current.children) == 3:
                    # A->(A)
                    rs = self._random_string_from_subtree(
                        current.children[1],
                        iteration_probability,
                        iteration_probability_decrease)
                elif len(current.children) == 5:
                    if current.children[2].symbol == '+':
                        # A->(A+A)
                        check_value = random.random()
                        if check_value < .5:
                            rs = self._random_string_from_subtree(
                                current.children[1],
                                iteration_probability,
                                iteration_probability_decrease)
                        else:
                            rs = self._random_string_from_subtree(
                                current.children[3],
                                iteration_probability,
                                iteration_probability_decrease)
                    else:
                        # A->(A.A)
                        rs = self._random_string_from_subtree(
                            current.children[1],
                            iteration_probability,
                            iteration_probability_decrease)
                        l1 = self._random_string_from_subtree(
                            current.children[3],
                            iteration_probability,
                            iteration_probability_decrease)
                        rs.extend(l1)
                else:
                    # A-> terminal
                    rs = self._random_string_from_subtree(
                        current.children[0],
                        iteration_probability,
                        iteration_probability_decrease)
            else:
                if len(current.children) == 3:
                    # B->(A)
                    rs = self._random_string_from_subtree(
                        current.children[1],
                        iteration_probability,
                        iteration_probability_decrease)
                elif len(current.children) == 5:
                    if current.children[2].symbol == '+':
                        # B->(A+A)
                        check_value = random.random()
                        if check_value < .5:
                            rs = self._random_string_from_subtree(
                                current.children[1],
                                iteration_probability,
                                iteration_probability_decrease)
                        else:
                            rs = self._random_string_from_subtree(
                                current.children[3],
                                iteration_probability,
                                iteration_probability_decrease)
                    else:
                        # B->(A.A)
                        rs = self._random_string_from_subtree(
                            current.children[1],
                            iteration_probability,
                            iteration_probability_decrease)
                        l1 = self._random_string_from_subtree(
                            current.children[3],
                            iteration_probability,
                            iteration_probability_decrease)
                        rs.extend(l1)
                else:
                    # B->terminal
                    rs = self._random_string_from_subtree(
                        current.children[0],
                        iteration_probability,
                        iteration_probability_decrease)
        return rs

    # def save(self, file):
    #     """Save a copy of the definition of this re in a json file."""
    #     d = vars(self).copy()
    #     with open(file+'.json', "w") as f:
    #         json.dump(d, f)
    
    @property
    def simple(self):
        stack = []
        for c in self.postfix:
            if c not in {'+','*','.'}:
                stack.append([c])
            elif c=='+':
                e = ['(']+stack[-2]+[c]+stack[-1]+[')']
                stack=stack[:-2]
                stack.append(e)
            elif c=='.':
                e = stack[-2]+stack[-1]
                stack=stack[:-2]
                stack.append(e)
            else:
                stack[-1]=stack[-1]+['*']
        return stack[-1]

    def __str__(self):
        """Return a string representation of the re."""
        # s = 'alphabet: {}\n'.format(', '.join(sorted(self.alphabet)))
        # if self.all_chars:
        #     s += 'expression: {}\n'.format(tools.Tools.print(self.expression))
        # else:
        #     s += 'expression: {}\n'.format(tools.Tools.print_tuple(self.expression))
        # return s
    
        """Return a string representation of the re."""
        s = 'alphabet: {}\n'.format(', '.join(sorted(self.alphabet)))
        if self.all_chars:
            s += 'expression: {}\n'.format(tools.Tools.print_tuple(self.simple, separator=''))
        else:
            s += 'expression: {}\n'.format(tools.Tools.print_tuple(self.simple, separator=' '))
        return s
