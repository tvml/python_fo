#!/usr/bin/env python3
"""Implementation of parsing through LL(1) parser."""

import tcs.base.base as base
import tcs.parse.base.parser_exceptions as pre

import tcs.grammar.cf.cf_grammar as cfg
import tcs.tools.tools as tools


class LL1_parser(base.Base):
    """A LL(1) parser."""

    def __init__(self, grammar):
        """Initialize."""
        self.parse_table = self.derive_parse_table(grammar)
        if self.parse_table is None:
            raise Exception('Grammar is not LL(1)')


    def parse(self, list_of_tokens):
        """Parse a sentence (string)"""
        # TODO

    def derive_parse_table(self, grammar):
        """Derive a LL(1) parse table."""
        # TODO
        return parse_table

    def __str__(self):
        """Return a string representation of the parser."""
        s = ''
        # TODO
        return s

    def __repr__(self):
        """Associate a description of the object to its identifier."""
        # TODO
        return None
