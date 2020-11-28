#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import computability.computability_exceptions as ce

"""Problem instance definitions."""

class PCP():
    
    def __init__(self, *, list_of_pairs):
        self.list_of_pairs = list_of_pairs
        self.n = len(list_of_pairs)
        
    def get_pair(self, i):
        if i<0 or i>=n:
            raise ce.PCPpair_exception()
        return self.list_of_pairs(i)