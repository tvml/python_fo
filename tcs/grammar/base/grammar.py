#!/usr/bin/env python3
"""Classes for working with all grammars."""

import tcs.base.base as base
import tcs.tools.tools as tools

import tcs.grammar.base.grammar_exceptions as ge

import tcs.grammar.base.derivation as der


class Grammar(base.Base):
    """
    A context sensitive grammar.

    Created by:
        Grammar(): definition provided as call parameters
        Grammar.load(file): definition provided in yaml file

    A Grammar is encoded as follows:
        - terminals are defined as strings
        - the set of terminals is a Python set of strings
        - nonterminals are defined as strings
        - the set of nonterminals is a Python set of strings
        - axiom is a string
        - productions is a Python dictionary where
            - keys are Python tuples of strings of length at least one
            - values are Python dicts where
                - keys are input symbols, including the empty string
                - values are sets of possibly empty Python tuples of strings
                Ab->a | AaA | '' is encoded as
                    productions[('A', 'b')]={('a',), ('A', 'a', 'A'), ()}
    """

    def __init__(self, terminals, non_terminals, axiom):
        """Initialize a complete automaton."""
        self.terminals = terminals.copy()
        self.non_terminals = non_terminals.copy()
        self.axiom = axiom
        self.all_chars = tools.Tools.all_chars(self.terminals.union(self.non_terminals))

    @staticmethod
    def build_productions(productions, non_terminals):
        """
        Derive internal representation of productions.

        Productions are coded as pairs of tuples of symbols.
        """
        prods = {}
        for left_part in productions.keys():
            lp = tools.Tools.tuple(left_part) if isinstance(left_part, str) \
                    else left_part
            # if isinstance(left_part, str):
            #     lp = Tools.tuple(left_part)
            # else:
            #     lp = left_part
            prods[lp] = set()
            for right_part in productions[left_part]:
                rp = tools.Tools.tuple(right_part) if isinstance(right_part, str) \
                    else right_part
                # if isinstance(right_part, str):
                #     rp = Tools.tuple(right_part)
                # else:
                #     rp = right_part
                prods[lp].add(rp)
        for nt in non_terminals:
            if nt not in productions.keys() and (nt,) not in productions.keys():
                prods[(nt,)] = set()
        return prods

    # @classmethod
    # def load(cls, file):
    #     """
    #     Load grammar from json file.

    #     Creates, as a result of the call cls.load(file),
    #     a grammar defined according to what is specified in the
    #     given file
    #     """
    #     with open(file+".json", "r") as read_file:
    #         data = json.load(read_file)
    #     return cls(terminals=data['terminals'],
    #                non_terminals=data['non_terminals'],
    #                axiom=data['axiom'],
    #                productions=data['productions'])

    def validate(self):
        """Return True if this RG is internally consistent."""
        self._validate_terminals()
        self._validate_non_terminals()
        self._validate_axiom()
        self._validate_productions()
        return True

    def _validate_terminals(self):
        """
        Raise an error if set of terminals is invalid.

        Checks that there exists at least one terminal symbol
        """
        # if len(self.terminals) == 0:
        #     raise ge.InvalidTerminalsError('no terminals defined in grammar')
        for t in self.terminals:
            if type(t) is not str:
                raise ge.InvalidTerminalsError('terminal {} in grammar is not a string'.
                                               format(t))

    def _validate_non_terminals(self):
        """
        Raise an error if set of non terminal is invalid.

        Checks that there exists at least one non terminal symbol and
        that the set of non terminals and the set of terminals are disjoint
        """
        if len(self.non_terminals) == 0:
            raise ge.InvalidNonTerminalsError(
                'no non terminals defined in grammar')
        if len(self.non_terminals.intersection(self.terminals)) > 0:
            raise ge.InvalidNonTerminalsError(
                'non terminals intersecting terminals in grammar')
        for nt in self.non_terminals:
            if type(nt) is not str:
                raise ge.InvalidTerminalsError(
                     'noncterminal {} in grammar is not a char'.format(nt))

    def _validate_axiom(self):
        """
        Raise an error if axiom is invalid.

        Checks that an axiom is defined and that it belongs to the set of
        non terminals
        """
        if self.axiom is None or len(self.axiom) == 0:
            raise ge.InvalidAxiomError('no axiom defined in grammar')
        if self.axiom not in self.non_terminals:
            raise ge.InvalidAxiomError('axiom not belonging to non terminals')

    def _validate_productions(self):
        """Check that all productions are valid."""
        for left_part, right_parts in self.productions.items():
            self._validate_left_part(left_part, right_parts)
            self._validate_right_parts(left_part, right_parts)

    def _validate_right_parts(self, left_part, right_parts):
        """Check that all right parts from a given left part are valid."""
        for right_part in right_parts:
            self._validate_right_part(left_part, right_part)

    def _validate_left_part(self, left_part, right_parts):
        """
        Raise an error if the left part of a production is invalid.

        Checks that the left part is not empty and contains at least one
        non terminal.
        """
        if left_part is None:
            raise ge.InvalidLeftPartError('no left part defined in production')
        if type(left_part) is not tuple:
            raise ge.InvalidLeftPartError('left part {} of production is invalid'.
                                          format(left_part))
        if set(left_part).isdisjoint(self.non_terminals.union({("S'",)})):
            raise ge.InvalidLeftPartError('no nonterminal in left part {} \
                                          of production'.
                                          format(left_part))

    def _validate_right_part(self, left_part, right_part):
        """
        Raise an error if the right part of a production is invalid.

        Checks that the right part is a (possibly empty) sequence of non
        terminals followed by a terminal
        """
        if right_part is None:
            raise ge.InvalidRightPartError(
                'no right part defined in production {} -> {}'.
                format(left_part, right_part))
        if type(right_part) is not tuple:
            raise ge.InvalidLeftPartError(
                     'right part {} of production is invalid'.
                     format(right_part))

    @property
    def left_parts(self):
        """Return all left parts of productions."""
        return {x for x in self.productions.keys()}

    def get_right_parts(self, left_part):
        """Return set of right parts produced by left part."""
        return self.productions.get(left_part)

    def random_derivation(self, n=100):
        """
        Return a derivation in grammar.

        Derivation is produced randomly.
        """
        derivation = der.Derivation.empty(self)
        for k in range(n):
            step = derivation.next_step()
            if step:
                derivation.append_step(step)
            else:
                break
        return derivation

    def random_left_derivation(self, n=100):
        """
        Return a leftmost derivation in grammar.

        Derivation is produced randomly.
        """
        derivation = der.Left_derivation.empty(self)
        for k in range(n):
            step = derivation.next_step()
            if step:
                derivation.append_step(step)
            else:
                break
        return derivation

    def random_right_derivation(self, n=100):
        """
        Return a leftmost derivation in grammar.

        Derivation is produced randomly.
        """
        derivation = der.Right_derivation.empty(self)
        for k in range(n):
            step = derivation.next_step()
            if step:
                derivation.append_step(step)
            else:
                break
        return derivation
    
    def derivations(self, depth=5):
        """
        Return all phrases and strings derived from derivations of length up to depth.
        """
        produced = []
        phrases = []
        produced.append(set())
        phrases.append(set())
        produced[0].add(self.axiom)
        for d in range(1,depth):
            produced.append(set())
            phrases.append(set())
            for phrase in list(produced[d-1]):
                for x in self.productions.keys():
                    lp = ''.join(x)
                    for ii in range(len(phrase)-len(lp)+1):
                        s3=phrase[ii:ii+len(lp)]
                        if s3==lp:
                            right_parts = self.productions[x]
                            for rp in right_parts:
                                s = ''
                                for j in range(ii):
                                    s+=phrase[j]
                                s+=''.join(rp)
                                for j in range(ii+len(lp),len(phrase)):
                                    s+=phrase[j]
                                if set(s).issubset(self.terminals):
                                    phrases[d].add(s)
                                else:
                                    produced[d].add(s)
        return phrases, produced
    
    def derived_strings(self, depth=5):
        """Prints all strings derived through derivations of length up to depth."""
        ph, pr = self.derivations(depth=depth)
        for i,x in enumerate(ph):
            print(str(i)+': ')
            for y in x:
                if len(y)>0:
                    print(y+' ', end = '')
                else:
                    print('_ ', end = '')
            print()
            
    def derived_phrases(self, depth=5):
        """Prints all phrases derived through derivations of length up to depth."""
        ph, pr = self.derivations(depth=depth)
        for i,x in enumerate(pr):
            print(str(i)+': ')
            for y in x:
                if len(y)>0:
                    print(y+' ', end = '')
                else:
                    print('_ ', end = '')
            print()

    def __str__(self):
        """Return a string representation of the grammar."""
        s = 'terminals: {}\n'.format(','.join(sorted(self.terminals)))
        s += 'non terminals: {}\n'.format(','.join(sorted(self.non_terminals)))
        s += 'axiom: {}\n'.format(self.axiom)
        s += 'productions\n'
        if self.all_chars:
            for left, rights in self.productions.items():
                s += '\t'+tools.Tools.print(left)+' -> '
                for right in rights:
                    s += ''+tools.Tools.print(right) + ' | '
                s = s[:-2]
                s += '\n'
            return s[:-2]
        else:
            for left, rights in self.productions.items():
                s += '\t'+tools.Tools.print_tuple(left)+' -> '
                for right in rights:
                    s += ''+tools.Tools.print_tuple(right) + ' | '
                s = s[:-2]
                s += '\n'
            return s[:-2]
