"""
parser_extensions.py — SOGA-specific helper methods for ANTLR Context classes.

Why this module exists
----------------------
ANTLR4 generates pure `*Parser.py`, `*Lexer.py`, `*Listener.py`, `*Visitor.py`
files. CLAUDE.md §8 forbids hand-editing them — yet the SOGA project
historically required several project-specific helpers attached to specific
Context subclasses (e.g. `SymvarsContext.getVar`, `TermContext.is_var`).
Before this module existed, those helpers were hand-injected into the
generated files. Any future regeneration via ANTLR jar silently overwrote
them, causing latent `AttributeError` and `TypeError 'NoneType' is not
callable` failures at runtime.

This module restores those helpers via monkey-patching at import time. The
generated `*Parser.py` files remain pure ANTLR output — safe to regenerate
without losing anything. Anywhere the parsers are invoked from runtime code
(libSOGAupdate.py, libSOGAtruncate.py, producecfg.py), import this module
at the top of the file so the patches are applied before the parser walks
a tree.

Methods restored
----------------
- SOGAParser.UniformContext.getText (override) — uniform(...) -> gm(...) text
- ASGMTParser.TermContext: is_var, is_const, getValue
- ASGMTParser.SymvarsContext: getVar
- ASGMTParser.IddContext: is_data, getValue
- TRUNCParser.IddContext: getVar, getValue
- TRUNCParser.VarContext: _getText
"""

import numpy as np

from SOGAParser import SOGAParser
from ASGMTParser import ASGMTParser
from TRUNCParser import TRUNCParser


# ---------------------------------------------------------------------------
# SOGAParser.UniformContext.getText (override — emits gm(...) text)
# ---------------------------------------------------------------------------

def _soga_uniform_get_text(self):
    """ converts string "uniform([a,b], K)" in "gm(pi, mu, sigma)" where gm
    is a Gaussian Mix with K components approximating the uniform. """
    a = float(self.list_().NUM()[0].getText())
    b = float(self.list_().NUM()[1].getText())
    N = int(self.NUM().getText())
    pi = [round(1.0 / N, 4)] * N
    mu = [round(a + i * (b - a) / N + ((b - a) / (2 * N)), 4) for i in range(N)]
    sigma = list([round((b - a) / (np.sqrt(12) * N), 4)] * N)
    print('gm(' + str(pi) + ',' + str(mu) + ',' + str(sigma) + ')')
    return 'gm(' + str(pi) + ',' + str(mu) + ',' + str(sigma) + ')'

SOGAParser.UniformContext.getText = _soga_uniform_get_text


# ---------------------------------------------------------------------------
# ASGMTParser.TermContext
# ---------------------------------------------------------------------------

def _asgmt_term_is_var(self, data):
    """ Returns 1 if term is a variable, 0 if it's a constant """
    if not self.NUM() is None:
        return False
    elif not self.symvars() is None:
        if not self.symvars().IDV() is None:
            return True
        elif not self.symvars().idd() is None:
            if self.symvars().idd().is_data(data):
                return False
            else:
                return True
    elif not self.gm() is None:
        return True

def _asgmt_term_is_const(self, data):
    return not self.is_var(data)

def _asgmt_term_get_value(self, data):
    if self.is_const(data):
        if not self.NUM() is None:
            return float(self.NUM().getText())
        elif not self.symvars() is None:
            return self.symvars().idd().getValue(data)
    else:
        raise("Calling getValue for a variable")

ASGMTParser.TermContext.is_var = _asgmt_term_is_var
ASGMTParser.TermContext.is_const = _asgmt_term_is_const
ASGMTParser.TermContext.getValue = _asgmt_term_get_value


# ---------------------------------------------------------------------------
# ASGMTParser.SymvarsContext
# ---------------------------------------------------------------------------

def _asgmt_symvars_get_var(self, data):
    if self.idd() is None:
        return self.getText()
    else:
        if self.idd().IDV(1) is None:
            return self.getText()
        else:
            data_idx = data[self.idd().IDV(1).getText()][0]
        return self.idd().IDV(0).getText() + '[' + str(data_idx) + ']'

ASGMTParser.SymvarsContext.getVar = _asgmt_symvars_get_var


# ---------------------------------------------------------------------------
# ASGMTParser.IddContext
# ---------------------------------------------------------------------------

def _asgmt_idd_is_data(self, data):
    if self.IDV(0).getText() in data.keys():
        return True
    else:
        return False

def _asgmt_idd_get_value(self, data):
    data_name = self.IDV(0).getText()
    if not self.NUM() is None:
        data_idx = int(self.NUM().getText())
    elif not self.IDV(1) is None:
        data_idx = data[self.IDV(1).getText()][0]
    return data[data_name][data_idx]

ASGMTParser.IddContext.is_data = _asgmt_idd_is_data
ASGMTParser.IddContext.getValue = _asgmt_idd_get_value


# ---------------------------------------------------------------------------
# TRUNCParser.IddContext
# ---------------------------------------------------------------------------

def _trunc_idd_get_var(self, data):
    if self.IDV(1) is None:
        return self.getText()
    else:
        data_idx = data[self.IDV(1).getText()][0]
        return self.IDV(0).getText() + '[' + str(data_idx) + ']'

def _trunc_idd_get_value(self, data):
    data_name = self.IDV(0).getText()
    if not self.NUM() is None:
        data_idx = int(self.NUM().getText())
    elif not self.IDV(1) is None:
        data_idx = data[self.IDV(1).getText()][0]
    return data[data_name][data_idx]

TRUNCParser.IddContext.getVar = _trunc_idd_get_var
TRUNCParser.IddContext.getValue = _trunc_idd_get_value


# ---------------------------------------------------------------------------
# TRUNCParser.VarContext
# ---------------------------------------------------------------------------

def _trunc_var_get_text(self, data):
    if not self.idd() is None:
        return self.idd().getVar(data)
    else:
        return self.getText()

TRUNCParser.VarContext._getText = _trunc_var_get_text
