# Generated from SOGA.g4 by ANTLR 4.10.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
import numpy as np
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,39,300,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        2,27,7,27,2,28,7,28,2,29,7,29,1,0,1,0,1,0,5,0,64,8,0,10,0,12,0,67,
        9,0,1,0,1,0,1,0,1,0,1,0,1,0,5,0,75,8,0,10,0,12,0,78,9,0,1,1,1,1,
        1,1,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1,3,3,3,95,8,3,1,
        4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,3,4,105,8,4,1,4,3,4,108,8,4,1,5,1,
        5,1,5,5,5,113,8,5,10,5,12,5,116,9,5,1,6,1,6,1,6,3,6,121,8,6,1,6,
        1,6,1,6,1,6,3,6,127,8,6,3,6,129,8,6,1,7,1,7,1,7,5,7,134,8,7,10,7,
        12,7,137,9,7,1,8,1,8,1,8,3,8,142,8,8,1,8,3,8,145,8,8,1,8,1,8,3,8,
        149,8,8,1,9,1,9,1,9,3,9,154,8,9,1,9,3,9,157,8,9,1,9,1,9,1,9,1,9,
        1,10,1,10,1,10,1,10,1,11,1,11,1,11,1,11,1,12,1,12,1,12,1,12,1,13,
        1,13,1,13,1,13,1,14,1,14,1,14,1,14,1,14,1,14,1,15,1,15,1,15,1,15,
        1,15,1,16,1,16,1,16,4,16,193,8,16,11,16,12,16,194,1,17,1,17,1,17,
        1,17,1,17,3,17,202,8,17,1,17,1,17,1,17,1,17,1,17,3,17,209,8,17,3,
        17,211,8,17,1,18,1,18,1,18,5,18,216,8,18,10,18,12,18,219,9,18,1,
        19,1,19,1,19,3,19,224,8,19,1,19,3,19,227,8,19,1,19,1,19,1,20,1,20,
        1,20,1,20,1,21,1,21,1,21,1,21,1,22,1,22,1,22,1,22,1,22,3,22,244,
        8,22,1,22,1,22,1,22,1,22,1,22,1,22,1,23,1,23,1,23,3,23,255,8,23,
        1,24,1,24,1,24,1,24,1,24,1,25,1,25,3,25,264,8,25,1,26,1,26,1,26,
        1,26,1,26,1,26,1,26,1,26,1,27,1,27,1,27,1,27,1,27,1,27,1,28,1,28,
        1,28,3,28,283,8,28,1,28,1,28,1,28,3,28,288,8,28,5,28,290,8,28,10,
        28,12,28,293,9,28,1,28,1,28,1,29,1,29,1,29,1,29,5,65,114,135,217,
        291,0,30,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,
        40,42,44,46,48,50,52,54,56,58,0,4,1,0,7,8,1,0,19,22,1,0,23,24,1,
        0,35,36,313,0,65,1,0,0,0,2,79,1,0,0,0,4,84,1,0,0,0,6,94,1,0,0,0,
        8,107,1,0,0,0,10,109,1,0,0,0,12,120,1,0,0,0,14,130,1,0,0,0,16,148,
        1,0,0,0,18,156,1,0,0,0,20,162,1,0,0,0,22,166,1,0,0,0,24,170,1,0,
        0,0,26,174,1,0,0,0,28,178,1,0,0,0,30,184,1,0,0,0,32,192,1,0,0,0,
        34,210,1,0,0,0,36,212,1,0,0,0,38,226,1,0,0,0,40,230,1,0,0,0,42,234,
        1,0,0,0,44,238,1,0,0,0,46,254,1,0,0,0,48,256,1,0,0,0,50,263,1,0,
        0,0,52,265,1,0,0,0,54,273,1,0,0,0,56,279,1,0,0,0,58,296,1,0,0,0,
        60,61,3,2,1,0,61,62,5,1,0,0,62,64,1,0,0,0,63,60,1,0,0,0,64,67,1,
        0,0,0,65,66,1,0,0,0,65,63,1,0,0,0,66,76,1,0,0,0,67,65,1,0,0,0,68,
        69,3,6,3,0,69,70,5,1,0,0,70,75,1,0,0,0,71,72,3,4,2,0,72,73,5,1,0,
        0,73,75,1,0,0,0,74,68,1,0,0,0,74,71,1,0,0,0,75,78,1,0,0,0,76,74,
        1,0,0,0,76,77,1,0,0,0,77,1,1,0,0,0,78,76,1,0,0,0,79,80,5,2,0,0,80,
        81,3,50,25,0,81,82,5,3,0,0,82,83,3,56,28,0,83,3,1,0,0,0,84,85,5,
        4,0,0,85,86,5,36,0,0,86,87,5,5,0,0,87,88,5,35,0,0,88,5,1,0,0,0,89,
        95,3,8,4,0,90,95,3,26,13,0,91,95,3,40,20,0,92,95,3,42,21,0,93,95,
        3,44,22,0,94,89,1,0,0,0,94,90,1,0,0,0,94,91,1,0,0,0,94,92,1,0,0,
        0,94,93,1,0,0,0,95,7,1,0,0,0,96,97,3,50,25,0,97,104,5,3,0,0,98,105,
        3,10,5,0,99,105,3,14,7,0,100,105,3,18,9,0,101,105,3,20,10,0,102,
        105,3,22,11,0,103,105,3,24,12,0,104,98,1,0,0,0,104,99,1,0,0,0,104,
        100,1,0,0,0,104,101,1,0,0,0,104,102,1,0,0,0,104,103,1,0,0,0,105,
        108,1,0,0,0,106,108,5,6,0,0,107,96,1,0,0,0,107,106,1,0,0,0,108,9,
        1,0,0,0,109,114,3,12,6,0,110,111,7,0,0,0,111,113,3,12,6,0,112,110,
        1,0,0,0,113,116,1,0,0,0,114,115,1,0,0,0,114,112,1,0,0,0,115,11,1,
        0,0,0,116,114,1,0,0,0,117,121,5,36,0,0,118,121,3,58,29,0,119,121,
        3,48,24,0,120,117,1,0,0,0,120,118,1,0,0,0,120,119,1,0,0,0,121,128,
        1,0,0,0,122,126,5,9,0,0,123,127,5,36,0,0,124,127,3,48,24,0,125,127,
        3,58,29,0,126,123,1,0,0,0,126,124,1,0,0,0,126,125,1,0,0,0,127,129,
        1,0,0,0,128,122,1,0,0,0,128,129,1,0,0,0,129,13,1,0,0,0,130,135,3,
        16,8,0,131,132,7,0,0,0,132,134,3,16,8,0,133,131,1,0,0,0,134,137,
        1,0,0,0,135,136,1,0,0,0,135,133,1,0,0,0,136,15,1,0,0,0,137,135,1,
        0,0,0,138,142,5,36,0,0,139,142,3,48,24,0,140,142,3,58,29,0,141,138,
        1,0,0,0,141,139,1,0,0,0,141,140,1,0,0,0,142,143,1,0,0,0,143,145,
        5,9,0,0,144,141,1,0,0,0,144,145,1,0,0,0,145,146,1,0,0,0,146,149,
        3,46,23,0,147,149,3,12,6,0,148,144,1,0,0,0,148,147,1,0,0,0,149,17,
        1,0,0,0,150,154,5,36,0,0,151,154,3,48,24,0,152,154,3,58,29,0,153,
        150,1,0,0,0,153,151,1,0,0,0,153,152,1,0,0,0,154,155,1,0,0,0,155,
        157,5,9,0,0,156,153,1,0,0,0,156,157,1,0,0,0,157,158,1,0,0,0,158,
        159,3,46,23,0,159,160,5,9,0,0,160,161,3,46,23,0,161,19,1,0,0,0,162,
        163,5,10,0,0,163,164,3,50,25,0,164,165,5,11,0,0,165,21,1,0,0,0,166,
        167,5,12,0,0,167,168,3,50,25,0,168,169,5,11,0,0,169,23,1,0,0,0,170,
        171,5,13,0,0,171,172,3,50,25,0,172,173,5,11,0,0,173,25,1,0,0,0,174,
        175,3,28,14,0,175,176,3,30,15,0,176,177,5,14,0,0,177,27,1,0,0,0,
        178,179,5,15,0,0,179,180,3,34,17,0,180,181,5,16,0,0,181,182,3,32,
        16,0,182,183,5,17,0,0,183,29,1,0,0,0,184,185,5,18,0,0,185,186,5,
        16,0,0,186,187,3,32,16,0,187,188,5,17,0,0,188,31,1,0,0,0,189,190,
        3,6,3,0,190,191,5,1,0,0,191,193,1,0,0,0,192,189,1,0,0,0,193,194,
        1,0,0,0,194,192,1,0,0,0,194,195,1,0,0,0,195,33,1,0,0,0,196,197,3,
        36,18,0,197,201,7,1,0,0,198,202,5,36,0,0,199,202,3,48,24,0,200,202,
        3,58,29,0,201,198,1,0,0,0,201,199,1,0,0,0,201,200,1,0,0,0,202,211,
        1,0,0,0,203,204,3,50,25,0,204,208,7,2,0,0,205,209,5,36,0,0,206,209,
        3,48,24,0,207,209,3,58,29,0,208,205,1,0,0,0,208,206,1,0,0,0,208,
        207,1,0,0,0,209,211,1,0,0,0,210,196,1,0,0,0,210,203,1,0,0,0,211,
        35,1,0,0,0,212,217,3,38,19,0,213,214,7,0,0,0,214,216,3,38,19,0,215,
        213,1,0,0,0,216,219,1,0,0,0,217,218,1,0,0,0,217,215,1,0,0,0,218,
        37,1,0,0,0,219,217,1,0,0,0,220,224,5,36,0,0,221,224,3,48,24,0,222,
        224,3,58,29,0,223,220,1,0,0,0,223,221,1,0,0,0,223,222,1,0,0,0,224,
        225,1,0,0,0,225,227,5,9,0,0,226,223,1,0,0,0,226,227,1,0,0,0,227,
        228,1,0,0,0,228,229,3,46,23,0,229,39,1,0,0,0,230,231,5,25,0,0,231,
        232,5,36,0,0,232,233,5,11,0,0,233,41,1,0,0,0,234,235,5,26,0,0,235,
        236,3,34,17,0,236,237,5,11,0,0,237,43,1,0,0,0,238,239,5,27,0,0,239,
        240,5,35,0,0,240,243,5,28,0,0,241,244,5,36,0,0,242,244,3,48,24,0,
        243,241,1,0,0,0,243,242,1,0,0,0,244,245,1,0,0,0,245,246,5,11,0,0,
        246,247,5,16,0,0,247,248,3,32,16,0,248,249,5,17,0,0,249,250,5,29,
        0,0,250,45,1,0,0,0,251,255,3,50,25,0,252,255,3,52,26,0,253,255,3,
        54,27,0,254,251,1,0,0,0,254,252,1,0,0,0,254,253,1,0,0,0,255,47,1,
        0,0,0,256,257,5,35,0,0,257,258,5,30,0,0,258,259,7,3,0,0,259,260,
        5,5,0,0,260,49,1,0,0,0,261,264,5,35,0,0,262,264,3,48,24,0,263,261,
        1,0,0,0,263,262,1,0,0,0,264,51,1,0,0,0,265,266,5,31,0,0,266,267,
        3,56,28,0,267,268,5,32,0,0,268,269,3,56,28,0,269,270,5,32,0,0,270,
        271,3,56,28,0,271,272,5,11,0,0,272,53,1,0,0,0,273,274,5,33,0,0,274,
        275,3,56,28,0,275,276,5,32,0,0,276,277,5,36,0,0,277,278,5,11,0,0,
        278,55,1,0,0,0,279,282,5,30,0,0,280,283,5,36,0,0,281,283,3,58,29,
        0,282,280,1,0,0,0,282,281,1,0,0,0,283,291,1,0,0,0,284,287,5,32,0,
        0,285,288,5,36,0,0,286,288,3,58,29,0,287,285,1,0,0,0,287,286,1,0,
        0,0,288,290,1,0,0,0,289,284,1,0,0,0,290,293,1,0,0,0,291,292,1,0,
        0,0,291,289,1,0,0,0,292,294,1,0,0,0,293,291,1,0,0,0,294,295,5,5,
        0,0,295,57,1,0,0,0,296,297,5,34,0,0,297,298,5,35,0,0,298,59,1,0,
        0,0,29,65,74,76,94,104,107,114,120,126,128,135,141,144,148,153,156,
        194,201,208,210,217,223,226,243,254,263,282,287,291
    ]

class SOGAParser ( Parser ):

    grammarFileName = "SOGA.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "';'", "'data'", "'='", "'array['", "']'", 
                     "'skip'", "'+'", "'-'", "'*'", "'exp('", "')'", "'sin('", 
                     "'cos('", "'end if'", "'if'", "'{'", "'}'", "'else'", 
                     "'<'", "'<='", "'>='", "'>'", "'=='", "'!='", "'prune('", 
                     "'observe('", "'for'", "'in range('", "'end for'", 
                     "'['", "'gm('", "','", "'uniform('", "'_'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "IDV", "NUM", 
                      "COMM", "WS", "DIGIT" ]

    RULE_progr = 0
    RULE_data = 1
    RULE_array = 2
    RULE_instr = 3
    RULE_assignment = 4
    RULE_const = 5
    RULE_const_term = 6
    RULE_add = 7
    RULE_add_term = 8
    RULE_mul = 9
    RULE_exp = 10
    RULE_sin = 11
    RULE_cos = 12
    RULE_conditional = 13
    RULE_ifclause = 14
    RULE_elseclause = 15
    RULE_block = 16
    RULE_bexpr = 17
    RULE_lexpr = 18
    RULE_monom = 19
    RULE_prune = 20
    RULE_observe = 21
    RULE_loop = 22
    RULE_vars = 23
    RULE_idd = 24
    RULE_symvars = 25
    RULE_gm = 26
    RULE_uniform = 27
    RULE_list = 28
    RULE_par = 29

    ruleNames =  [ "progr", "data", "array", "instr", "assignment", "const", 
                   "const_term", "add", "add_term", "mul", "exp", "sin", 
                   "cos", "conditional", "ifclause", "elseclause", "block", 
                   "bexpr", "lexpr", "monom", "prune", "observe", "loop", 
                   "vars", "idd", "symvars", "gm", "uniform", "list", "par" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    T__11=12
    T__12=13
    T__13=14
    T__14=15
    T__15=16
    T__16=17
    T__17=18
    T__18=19
    T__19=20
    T__20=21
    T__21=22
    T__22=23
    T__23=24
    T__24=25
    T__25=26
    T__26=27
    T__27=28
    T__28=29
    T__29=30
    T__30=31
    T__31=32
    T__32=33
    T__33=34
    IDV=35
    NUM=36
    COMM=37
    WS=38
    DIGIT=39

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgrContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def data(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.DataContext)
            else:
                return self.getTypedRuleContext(SOGAParser.DataContext,i)


        def instr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.InstrContext)
            else:
                return self.getTypedRuleContext(SOGAParser.InstrContext,i)


        def array(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.ArrayContext)
            else:
                return self.getTypedRuleContext(SOGAParser.ArrayContext,i)


        def getRuleIndex(self):
            return SOGAParser.RULE_progr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProgr" ):
                listener.enterProgr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProgr" ):
                listener.exitProgr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProgr" ):
                return visitor.visitProgr(self)
            else:
                return visitor.visitChildren(self)




    def progr(self):

        localctx = SOGAParser.ProgrContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_progr)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 65
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,0,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 60
                    self.data()
                    self.state = 61
                    self.match(SOGAParser.T__0) 
                self.state = 67
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,0,self._ctx)

            self.state = 76
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << SOGAParser.T__3) | (1 << SOGAParser.T__5) | (1 << SOGAParser.T__14) | (1 << SOGAParser.T__24) | (1 << SOGAParser.T__25) | (1 << SOGAParser.T__26) | (1 << SOGAParser.IDV))) != 0):
                self.state = 74
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.T__5, SOGAParser.T__14, SOGAParser.T__24, SOGAParser.T__25, SOGAParser.T__26, SOGAParser.IDV]:
                    self.state = 68
                    self.instr()
                    self.state = 69
                    self.match(SOGAParser.T__0)
                    pass
                elif token in [SOGAParser.T__3]:
                    self.state = 71
                    self.array()
                    self.state = 72
                    self.match(SOGAParser.T__0)
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 78
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DataContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def symvars(self):
            return self.getTypedRuleContext(SOGAParser.SymvarsContext,0)


        def list_(self):
            return self.getTypedRuleContext(SOGAParser.ListContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_data

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterData" ):
                listener.enterData(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitData" ):
                listener.exitData(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitData" ):
                return visitor.visitData(self)
            else:
                return visitor.visitChildren(self)




    def data(self):

        localctx = SOGAParser.DataContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_data)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 79
            self.match(SOGAParser.T__1)
            self.state = 80
            self.symvars()
            self.state = 81
            self.match(SOGAParser.T__2)
            self.state = 82
            self.list_()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ArrayContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUM(self):
            return self.getToken(SOGAParser.NUM, 0)

        def IDV(self):
            return self.getToken(SOGAParser.IDV, 0)

        def getRuleIndex(self):
            return SOGAParser.RULE_array

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterArray" ):
                listener.enterArray(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitArray" ):
                listener.exitArray(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitArray" ):
                return visitor.visitArray(self)
            else:
                return visitor.visitChildren(self)




    def array(self):

        localctx = SOGAParser.ArrayContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_array)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 84
            self.match(SOGAParser.T__3)
            self.state = 85
            self.match(SOGAParser.NUM)
            self.state = 86
            self.match(SOGAParser.T__4)
            self.state = 87
            self.match(SOGAParser.IDV)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class InstrContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def assignment(self):
            return self.getTypedRuleContext(SOGAParser.AssignmentContext,0)


        def conditional(self):
            return self.getTypedRuleContext(SOGAParser.ConditionalContext,0)


        def prune(self):
            return self.getTypedRuleContext(SOGAParser.PruneContext,0)


        def observe(self):
            return self.getTypedRuleContext(SOGAParser.ObserveContext,0)


        def loop(self):
            return self.getTypedRuleContext(SOGAParser.LoopContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_instr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInstr" ):
                listener.enterInstr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInstr" ):
                listener.exitInstr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitInstr" ):
                return visitor.visitInstr(self)
            else:
                return visitor.visitChildren(self)




    def instr(self):

        localctx = SOGAParser.InstrContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_instr)
        try:
            self.state = 94
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.T__5, SOGAParser.IDV]:
                self.enterOuterAlt(localctx, 1)
                self.state = 89
                self.assignment()
                pass
            elif token in [SOGAParser.T__14]:
                self.enterOuterAlt(localctx, 2)
                self.state = 90
                self.conditional()
                pass
            elif token in [SOGAParser.T__24]:
                self.enterOuterAlt(localctx, 3)
                self.state = 91
                self.prune()
                pass
            elif token in [SOGAParser.T__25]:
                self.enterOuterAlt(localctx, 4)
                self.state = 92
                self.observe()
                pass
            elif token in [SOGAParser.T__26]:
                self.enterOuterAlt(localctx, 5)
                self.state = 93
                self.loop()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AssignmentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def symvars(self):
            return self.getTypedRuleContext(SOGAParser.SymvarsContext,0)


        def const(self):
            return self.getTypedRuleContext(SOGAParser.ConstContext,0)


        def add(self):
            return self.getTypedRuleContext(SOGAParser.AddContext,0)


        def mul(self):
            return self.getTypedRuleContext(SOGAParser.MulContext,0)


        def exp(self):
            return self.getTypedRuleContext(SOGAParser.ExpContext,0)


        def sin(self):
            return self.getTypedRuleContext(SOGAParser.SinContext,0)


        def cos(self):
            return self.getTypedRuleContext(SOGAParser.CosContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_assignment

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAssignment" ):
                listener.enterAssignment(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAssignment" ):
                listener.exitAssignment(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAssignment" ):
                return visitor.visitAssignment(self)
            else:
                return visitor.visitChildren(self)




    def assignment(self):

        localctx = SOGAParser.AssignmentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_assignment)
        try:
            self.state = 107
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.IDV]:
                self.enterOuterAlt(localctx, 1)
                self.state = 96
                self.symvars()
                self.state = 97
                self.match(SOGAParser.T__2)
                self.state = 104
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,4,self._ctx)
                if la_ == 1:
                    self.state = 98
                    self.const()
                    pass

                elif la_ == 2:
                    self.state = 99
                    self.add()
                    pass

                elif la_ == 3:
                    self.state = 100
                    self.mul()
                    pass

                elif la_ == 4:
                    self.state = 101
                    self.exp()
                    pass

                elif la_ == 5:
                    self.state = 102
                    self.sin()
                    pass

                elif la_ == 6:
                    self.state = 103
                    self.cos()
                    pass


                pass
            elif token in [SOGAParser.T__5]:
                self.enterOuterAlt(localctx, 2)
                self.state = 106
                self.match(SOGAParser.T__5)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ConstContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def const_term(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.Const_termContext)
            else:
                return self.getTypedRuleContext(SOGAParser.Const_termContext,i)


        def getRuleIndex(self):
            return SOGAParser.RULE_const

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterConst" ):
                listener.enterConst(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitConst" ):
                listener.exitConst(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitConst" ):
                return visitor.visitConst(self)
            else:
                return visitor.visitChildren(self)




    def const(self):

        localctx = SOGAParser.ConstContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_const)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 109
            self.const_term()
            self.state = 114
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,6,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 110
                    _la = self._input.LA(1)
                    if not(_la==SOGAParser.T__6 or _la==SOGAParser.T__7):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 111
                    self.const_term() 
                self.state = 116
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,6,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Const_termContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUM(self, i:int=None):
            if i is None:
                return self.getTokens(SOGAParser.NUM)
            else:
                return self.getToken(SOGAParser.NUM, i)

        def par(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.ParContext)
            else:
                return self.getTypedRuleContext(SOGAParser.ParContext,i)


        def idd(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.IddContext)
            else:
                return self.getTypedRuleContext(SOGAParser.IddContext,i)


        def getRuleIndex(self):
            return SOGAParser.RULE_const_term

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterConst_term" ):
                listener.enterConst_term(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitConst_term" ):
                listener.exitConst_term(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitConst_term" ):
                return visitor.visitConst_term(self)
            else:
                return visitor.visitChildren(self)




    def const_term(self):

        localctx = SOGAParser.Const_termContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_const_term)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 120
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.NUM]:
                self.state = 117
                self.match(SOGAParser.NUM)
                pass
            elif token in [SOGAParser.T__33]:
                self.state = 118
                self.par()
                pass
            elif token in [SOGAParser.IDV]:
                self.state = 119
                self.idd()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 128
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==SOGAParser.T__8:
                self.state = 122
                self.match(SOGAParser.T__8)
                self.state = 126
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 123
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 124
                    self.idd()
                    pass
                elif token in [SOGAParser.T__33]:
                    self.state = 125
                    self.par()
                    pass
                else:
                    raise NoViableAltException(self)



        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AddContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def add_term(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.Add_termContext)
            else:
                return self.getTypedRuleContext(SOGAParser.Add_termContext,i)


        def getRuleIndex(self):
            return SOGAParser.RULE_add

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAdd" ):
                listener.enterAdd(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAdd" ):
                listener.exitAdd(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAdd" ):
                return visitor.visitAdd(self)
            else:
                return visitor.visitChildren(self)




    def add(self):

        localctx = SOGAParser.AddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_add)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 130
            self.add_term()
            self.state = 135
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,10,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 131
                    _la = self._input.LA(1)
                    if not(_la==SOGAParser.T__6 or _la==SOGAParser.T__7):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 132
                    self.add_term() 
                self.state = 137
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,10,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Add_termContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def vars_(self):
            return self.getTypedRuleContext(SOGAParser.VarsContext,0)


        def NUM(self):
            return self.getToken(SOGAParser.NUM, 0)

        def idd(self):
            return self.getTypedRuleContext(SOGAParser.IddContext,0)


        def par(self):
            return self.getTypedRuleContext(SOGAParser.ParContext,0)


        def const_term(self):
            return self.getTypedRuleContext(SOGAParser.Const_termContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_add_term

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAdd_term" ):
                listener.enterAdd_term(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAdd_term" ):
                listener.exitAdd_term(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAdd_term" ):
                return visitor.visitAdd_term(self)
            else:
                return visitor.visitChildren(self)




    def add_term(self):

        localctx = SOGAParser.Add_termContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_add_term)
        try:
            self.state = 148
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,13,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 144
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,12,self._ctx)
                if la_ == 1:
                    self.state = 141
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [SOGAParser.NUM]:
                        self.state = 138
                        self.match(SOGAParser.NUM)
                        pass
                    elif token in [SOGAParser.IDV]:
                        self.state = 139
                        self.idd()
                        pass
                    elif token in [SOGAParser.T__33]:
                        self.state = 140
                        self.par()
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 143
                    self.match(SOGAParser.T__8)


                self.state = 146
                self.vars_()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 147
                self.const_term()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MulContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def vars_(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.VarsContext)
            else:
                return self.getTypedRuleContext(SOGAParser.VarsContext,i)


        def NUM(self):
            return self.getToken(SOGAParser.NUM, 0)

        def idd(self):
            return self.getTypedRuleContext(SOGAParser.IddContext,0)


        def par(self):
            return self.getTypedRuleContext(SOGAParser.ParContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_mul

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMul" ):
                listener.enterMul(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMul" ):
                listener.exitMul(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMul" ):
                return visitor.visitMul(self)
            else:
                return visitor.visitChildren(self)




    def mul(self):

        localctx = SOGAParser.MulContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_mul)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 156
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,15,self._ctx)
            if la_ == 1:
                self.state = 153
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 150
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 151
                    self.idd()
                    pass
                elif token in [SOGAParser.T__33]:
                    self.state = 152
                    self.par()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 155
                self.match(SOGAParser.T__8)


            self.state = 158
            self.vars_()
            self.state = 159
            self.match(SOGAParser.T__8)
            self.state = 160
            self.vars_()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExpContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def symvars(self):
            return self.getTypedRuleContext(SOGAParser.SymvarsContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_exp

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExp" ):
                listener.enterExp(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExp" ):
                listener.exitExp(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExp" ):
                return visitor.visitExp(self)
            else:
                return visitor.visitChildren(self)




    def exp(self):

        localctx = SOGAParser.ExpContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_exp)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 162
            self.match(SOGAParser.T__9)
            self.state = 163
            self.symvars()
            self.state = 164
            self.match(SOGAParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SinContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def symvars(self):
            return self.getTypedRuleContext(SOGAParser.SymvarsContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_sin

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSin" ):
                listener.enterSin(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSin" ):
                listener.exitSin(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSin" ):
                return visitor.visitSin(self)
            else:
                return visitor.visitChildren(self)




    def sin(self):

        localctx = SOGAParser.SinContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_sin)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 166
            self.match(SOGAParser.T__11)
            self.state = 167
            self.symvars()
            self.state = 168
            self.match(SOGAParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CosContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def symvars(self):
            return self.getTypedRuleContext(SOGAParser.SymvarsContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_cos

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCos" ):
                listener.enterCos(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCos" ):
                listener.exitCos(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCos" ):
                return visitor.visitCos(self)
            else:
                return visitor.visitChildren(self)




    def cos(self):

        localctx = SOGAParser.CosContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_cos)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 170
            self.match(SOGAParser.T__12)
            self.state = 171
            self.symvars()
            self.state = 172
            self.match(SOGAParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ConditionalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ifclause(self):
            return self.getTypedRuleContext(SOGAParser.IfclauseContext,0)


        def elseclause(self):
            return self.getTypedRuleContext(SOGAParser.ElseclauseContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_conditional

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterConditional" ):
                listener.enterConditional(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitConditional" ):
                listener.exitConditional(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitConditional" ):
                return visitor.visitConditional(self)
            else:
                return visitor.visitChildren(self)




    def conditional(self):

        localctx = SOGAParser.ConditionalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_conditional)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 174
            self.ifclause()
            self.state = 175
            self.elseclause()
            self.state = 176
            self.match(SOGAParser.T__13)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IfclauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def bexpr(self):
            return self.getTypedRuleContext(SOGAParser.BexprContext,0)


        def block(self):
            return self.getTypedRuleContext(SOGAParser.BlockContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_ifclause

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIfclause" ):
                listener.enterIfclause(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIfclause" ):
                listener.exitIfclause(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIfclause" ):
                return visitor.visitIfclause(self)
            else:
                return visitor.visitChildren(self)




    def ifclause(self):

        localctx = SOGAParser.IfclauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_ifclause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 178
            self.match(SOGAParser.T__14)
            self.state = 179
            self.bexpr()
            self.state = 180
            self.match(SOGAParser.T__15)
            self.state = 181
            self.block()
            self.state = 182
            self.match(SOGAParser.T__16)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ElseclauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def block(self):
            return self.getTypedRuleContext(SOGAParser.BlockContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_elseclause

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterElseclause" ):
                listener.enterElseclause(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitElseclause" ):
                listener.exitElseclause(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitElseclause" ):
                return visitor.visitElseclause(self)
            else:
                return visitor.visitChildren(self)




    def elseclause(self):

        localctx = SOGAParser.ElseclauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_elseclause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 184
            self.match(SOGAParser.T__17)
            self.state = 185
            self.match(SOGAParser.T__15)
            self.state = 186
            self.block()
            self.state = 187
            self.match(SOGAParser.T__16)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BlockContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def instr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.InstrContext)
            else:
                return self.getTypedRuleContext(SOGAParser.InstrContext,i)


        def getRuleIndex(self):
            return SOGAParser.RULE_block

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBlock" ):
                listener.enterBlock(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBlock" ):
                listener.exitBlock(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBlock" ):
                return visitor.visitBlock(self)
            else:
                return visitor.visitChildren(self)




    def block(self):

        localctx = SOGAParser.BlockContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_block)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 192 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 189
                self.instr()
                self.state = 190
                self.match(SOGAParser.T__0)
                self.state = 194 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not ((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << SOGAParser.T__5) | (1 << SOGAParser.T__14) | (1 << SOGAParser.T__24) | (1 << SOGAParser.T__25) | (1 << SOGAParser.T__26) | (1 << SOGAParser.IDV))) != 0)):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def lexpr(self):
            return self.getTypedRuleContext(SOGAParser.LexprContext,0)


        def NUM(self):
            return self.getToken(SOGAParser.NUM, 0)

        def idd(self):
            return self.getTypedRuleContext(SOGAParser.IddContext,0)


        def par(self):
            return self.getTypedRuleContext(SOGAParser.ParContext,0)


        def symvars(self):
            return self.getTypedRuleContext(SOGAParser.SymvarsContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_bexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBexpr" ):
                listener.enterBexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBexpr" ):
                listener.exitBexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBexpr" ):
                return visitor.visitBexpr(self)
            else:
                return visitor.visitChildren(self)




    def bexpr(self):

        localctx = SOGAParser.BexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_bexpr)
        self._la = 0 # Token type
        try:
            self.state = 210
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,19,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 196
                self.lexpr()
                self.state = 197
                _la = self._input.LA(1)
                if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << SOGAParser.T__18) | (1 << SOGAParser.T__19) | (1 << SOGAParser.T__20) | (1 << SOGAParser.T__21))) != 0)):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 201
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 198
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 199
                    self.idd()
                    pass
                elif token in [SOGAParser.T__33]:
                    self.state = 200
                    self.par()
                    pass
                else:
                    raise NoViableAltException(self)

                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 203
                self.symvars()
                self.state = 204
                _la = self._input.LA(1)
                if not(_la==SOGAParser.T__22 or _la==SOGAParser.T__23):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 208
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 205
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 206
                    self.idd()
                    pass
                elif token in [SOGAParser.T__33]:
                    self.state = 207
                    self.par()
                    pass
                else:
                    raise NoViableAltException(self)

                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def monom(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.MonomContext)
            else:
                return self.getTypedRuleContext(SOGAParser.MonomContext,i)


        def getRuleIndex(self):
            return SOGAParser.RULE_lexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLexpr" ):
                listener.enterLexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLexpr" ):
                listener.exitLexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLexpr" ):
                return visitor.visitLexpr(self)
            else:
                return visitor.visitChildren(self)




    def lexpr(self):

        localctx = SOGAParser.LexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_lexpr)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 212
            self.monom()
            self.state = 217
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,20,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 213
                    _la = self._input.LA(1)
                    if not(_la==SOGAParser.T__6 or _la==SOGAParser.T__7):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 214
                    self.monom() 
                self.state = 219
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,20,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MonomContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def vars_(self):
            return self.getTypedRuleContext(SOGAParser.VarsContext,0)


        def NUM(self):
            return self.getToken(SOGAParser.NUM, 0)

        def idd(self):
            return self.getTypedRuleContext(SOGAParser.IddContext,0)


        def par(self):
            return self.getTypedRuleContext(SOGAParser.ParContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_monom

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMonom" ):
                listener.enterMonom(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMonom" ):
                listener.exitMonom(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMonom" ):
                return visitor.visitMonom(self)
            else:
                return visitor.visitChildren(self)




    def monom(self):

        localctx = SOGAParser.MonomContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_monom)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 226
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,22,self._ctx)
            if la_ == 1:
                self.state = 223
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 220
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 221
                    self.idd()
                    pass
                elif token in [SOGAParser.T__33]:
                    self.state = 222
                    self.par()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 225
                self.match(SOGAParser.T__8)


            self.state = 228
            self.vars_()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PruneContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUM(self):
            return self.getToken(SOGAParser.NUM, 0)

        def getRuleIndex(self):
            return SOGAParser.RULE_prune

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPrune" ):
                listener.enterPrune(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPrune" ):
                listener.exitPrune(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPrune" ):
                return visitor.visitPrune(self)
            else:
                return visitor.visitChildren(self)




    def prune(self):

        localctx = SOGAParser.PruneContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_prune)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 230
            self.match(SOGAParser.T__24)
            self.state = 231
            self.match(SOGAParser.NUM)
            self.state = 232
            self.match(SOGAParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ObserveContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def bexpr(self):
            return self.getTypedRuleContext(SOGAParser.BexprContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_observe

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterObserve" ):
                listener.enterObserve(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitObserve" ):
                listener.exitObserve(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitObserve" ):
                return visitor.visitObserve(self)
            else:
                return visitor.visitChildren(self)




    def observe(self):

        localctx = SOGAParser.ObserveContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_observe)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 234
            self.match(SOGAParser.T__25)
            self.state = 235
            self.bexpr()
            self.state = 236
            self.match(SOGAParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LoopContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self):
            return self.getToken(SOGAParser.IDV, 0)

        def block(self):
            return self.getTypedRuleContext(SOGAParser.BlockContext,0)


        def NUM(self):
            return self.getToken(SOGAParser.NUM, 0)

        def idd(self):
            return self.getTypedRuleContext(SOGAParser.IddContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_loop

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLoop" ):
                listener.enterLoop(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLoop" ):
                listener.exitLoop(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLoop" ):
                return visitor.visitLoop(self)
            else:
                return visitor.visitChildren(self)




    def loop(self):

        localctx = SOGAParser.LoopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_loop)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 238
            self.match(SOGAParser.T__26)
            self.state = 239
            self.match(SOGAParser.IDV)
            self.state = 240
            self.match(SOGAParser.T__27)
            self.state = 243
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.NUM]:
                self.state = 241
                self.match(SOGAParser.NUM)
                pass
            elif token in [SOGAParser.IDV]:
                self.state = 242
                self.idd()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 245
            self.match(SOGAParser.T__10)
            self.state = 246
            self.match(SOGAParser.T__15)
            self.state = 247
            self.block()
            self.state = 248
            self.match(SOGAParser.T__16)
            self.state = 249
            self.match(SOGAParser.T__28)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class VarsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def symvars(self):
            return self.getTypedRuleContext(SOGAParser.SymvarsContext,0)


        def gm(self):
            return self.getTypedRuleContext(SOGAParser.GmContext,0)


        def uniform(self):
            return self.getTypedRuleContext(SOGAParser.UniformContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_vars

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVars" ):
                listener.enterVars(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVars" ):
                listener.exitVars(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVars" ):
                return visitor.visitVars(self)
            else:
                return visitor.visitChildren(self)




    def vars_(self):

        localctx = SOGAParser.VarsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_vars)
        try:
            self.state = 254
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.IDV]:
                self.enterOuterAlt(localctx, 1)
                self.state = 251
                self.symvars()
                pass
            elif token in [SOGAParser.T__30]:
                self.enterOuterAlt(localctx, 2)
                self.state = 252
                self.gm()
                pass
            elif token in [SOGAParser.T__32]:
                self.enterOuterAlt(localctx, 3)
                self.state = 253
                self.uniform()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IddContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self, i:int=None):
            if i is None:
                return self.getTokens(SOGAParser.IDV)
            else:
                return self.getToken(SOGAParser.IDV, i)

        def NUM(self):
            return self.getToken(SOGAParser.NUM, 0)

        def getRuleIndex(self):
            return SOGAParser.RULE_idd

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIdd" ):
                listener.enterIdd(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIdd" ):
                listener.exitIdd(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIdd" ):
                return visitor.visitIdd(self)
            else:
                return visitor.visitChildren(self)




    def idd(self):

        localctx = SOGAParser.IddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_idd)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 256
            self.match(SOGAParser.IDV)
            self.state = 257
            self.match(SOGAParser.T__29)
            self.state = 258
            _la = self._input.LA(1)
            if not(_la==SOGAParser.IDV or _la==SOGAParser.NUM):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 259
            self.match(SOGAParser.T__4)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SymvarsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self):
            return self.getToken(SOGAParser.IDV, 0)

        def idd(self):
            return self.getTypedRuleContext(SOGAParser.IddContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_symvars

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSymvars" ):
                listener.enterSymvars(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSymvars" ):
                listener.exitSymvars(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSymvars" ):
                return visitor.visitSymvars(self)
            else:
                return visitor.visitChildren(self)




    def symvars(self):

        localctx = SOGAParser.SymvarsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_symvars)
        try:
            self.state = 263
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,25,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 261
                self.match(SOGAParser.IDV)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 262
                self.idd()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GmContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def list_(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.ListContext)
            else:
                return self.getTypedRuleContext(SOGAParser.ListContext,i)


        def getRuleIndex(self):
            return SOGAParser.RULE_gm

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterGm" ):
                listener.enterGm(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitGm" ):
                listener.exitGm(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitGm" ):
                return visitor.visitGm(self)
            else:
                return visitor.visitChildren(self)




    def gm(self):

        localctx = SOGAParser.GmContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_gm)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 265
            self.match(SOGAParser.T__30)
            self.state = 266
            self.list_()
            self.state = 267
            self.match(SOGAParser.T__31)
            self.state = 268
            self.list_()
            self.state = 269
            self.match(SOGAParser.T__31)
            self.state = 270
            self.list_()
            self.state = 271
            self.match(SOGAParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class UniformContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def list_(self):
            return self.getTypedRuleContext(SOGAParser.ListContext,0)


        def NUM(self):
            return self.getToken(SOGAParser.NUM, 0)

        def getRuleIndex(self):
            return SOGAParser.RULE_uniform

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterUniform" ):
                listener.enterUniform(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitUniform" ):
                listener.exitUniform(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitUniform" ):
                return visitor.visitUniform(self)
            else:
                return visitor.visitChildren(self)

        def getText(self):
            """ converts string "uniform([a,b], K)" in "gm(pi, mu, sigma)" where gm is a Gaussian Mix with K component approximating the uniform"""
            a = float(self.list_().NUM()[0].getText())
            b = float(self.list_().NUM()[1].getText())
            N = int(self.NUM().getText())
            pi = [round(1.0/N,4)]*N
            mu = [round(a+i*(b-a)/N+((b-a)/(2*N)),4) for i in range(N)]
            sigma = list([round((b-a)/(np.sqrt(12)*N),4)]*N)
            return 'gm('+str(pi)+','+str(mu)+','+str(sigma)+')'
  

    def uniform(self):

        localctx = SOGAParser.UniformContext(self, self._ctx, self.state)
        self.enterRule(localctx, 54, self.RULE_uniform)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 273
            self.match(SOGAParser.T__32)
            self.state = 274
            self.list_()
            self.state = 275
            self.match(SOGAParser.T__31)
            self.state = 276
            self.match(SOGAParser.NUM)
            self.state = 277
            self.match(SOGAParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUM(self, i:int=None):
            if i is None:
                return self.getTokens(SOGAParser.NUM)
            else:
                return self.getToken(SOGAParser.NUM, i)

        def par(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.ParContext)
            else:
                return self.getTypedRuleContext(SOGAParser.ParContext,i)


        def getRuleIndex(self):
            return SOGAParser.RULE_list

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterList" ):
                listener.enterList(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitList" ):
                listener.exitList(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitList" ):
                return visitor.visitList(self)
            else:
                return visitor.visitChildren(self)




    def list_(self):

        localctx = SOGAParser.ListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 56, self.RULE_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 279
            self.match(SOGAParser.T__29)
            self.state = 282
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.NUM]:
                self.state = 280
                self.match(SOGAParser.NUM)
                pass
            elif token in [SOGAParser.T__33]:
                self.state = 281
                self.par()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 291
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,28,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 284
                    self.match(SOGAParser.T__31)
                    self.state = 287
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [SOGAParser.NUM]:
                        self.state = 285
                        self.match(SOGAParser.NUM)
                        pass
                    elif token in [SOGAParser.T__33]:
                        self.state = 286
                        self.par()
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 293
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,28,self._ctx)

            self.state = 294
            self.match(SOGAParser.T__4)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ParContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self):
            return self.getToken(SOGAParser.IDV, 0)

        def getRuleIndex(self):
            return SOGAParser.RULE_par

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPar" ):
                listener.enterPar(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPar" ):
                listener.exitPar(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPar" ):
                return visitor.visitPar(self)
            else:
                return visitor.visitChildren(self)




    def par(self):

        localctx = SOGAParser.ParContext(self, self._ctx, self.state)
        self.enterRule(localctx, 58, self.RULE_par)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 296
            self.match(SOGAParser.T__33)
            self.state = 297
            self.match(SOGAParser.IDV)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





