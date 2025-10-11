# Generated from SOGA.g4 by ANTLR 4.10.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import numpy as np
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,41,310,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        2,27,7,27,2,28,7,28,2,29,7,29,2,30,7,30,1,0,1,0,1,0,5,0,66,8,0,10,
        0,12,0,69,9,0,1,0,1,0,1,0,1,0,1,0,1,0,5,0,77,8,0,10,0,12,0,80,9,
        0,1,1,1,1,1,1,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1,3,1,
        3,3,3,98,8,3,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,3,4,108,8,4,1,4,3,4,
        111,8,4,1,5,1,5,1,5,5,5,116,8,5,10,5,12,5,119,9,5,1,6,1,6,1,6,3,
        6,124,8,6,1,6,1,6,1,6,1,6,3,6,130,8,6,3,6,132,8,6,1,7,1,7,1,7,5,
        7,137,8,7,10,7,12,7,140,9,7,1,8,1,8,1,8,3,8,145,8,8,1,8,3,8,148,
        8,8,1,8,1,8,3,8,152,8,8,1,9,1,9,1,9,3,9,157,8,9,1,9,3,9,160,8,9,
        1,9,1,9,1,9,1,9,1,10,1,10,1,10,1,10,1,11,1,11,1,11,1,11,1,12,1,12,
        1,12,1,12,1,13,1,13,1,13,1,13,1,14,1,14,1,14,1,14,1,14,1,14,1,15,
        1,15,1,15,1,15,1,15,1,16,1,16,1,16,4,16,196,8,16,11,16,12,16,197,
        1,17,1,17,1,17,1,17,1,17,3,17,205,8,17,1,17,1,17,1,17,1,17,1,17,
        3,17,212,8,17,3,17,214,8,17,1,18,1,18,1,18,5,18,219,8,18,10,18,12,
        18,222,9,18,1,19,1,19,1,19,3,19,227,8,19,1,19,3,19,230,8,19,1,19,
        1,19,1,20,1,20,1,20,1,20,1,21,1,21,1,21,1,21,1,22,1,22,1,22,1,22,
        1,22,3,22,247,8,22,1,22,1,22,1,22,1,22,1,22,1,22,1,23,1,23,1,23,
        1,23,1,23,1,23,1,23,1,24,1,24,1,24,3,24,265,8,24,1,25,1,25,1,25,
        1,25,1,25,1,26,1,26,3,26,274,8,26,1,27,1,27,1,27,1,27,1,27,1,27,
        1,27,1,27,1,28,1,28,1,28,1,28,1,28,1,28,1,29,1,29,1,29,3,29,293,
        8,29,1,29,1,29,1,29,3,29,298,8,29,5,29,300,8,29,10,29,12,29,303,
        9,29,1,29,1,29,1,30,1,30,1,30,1,30,5,67,117,138,220,301,0,31,0,2,
        4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,
        50,52,54,56,58,60,0,4,1,0,7,8,1,0,19,22,1,0,23,24,1,0,37,38,323,
        0,67,1,0,0,0,2,81,1,0,0,0,4,86,1,0,0,0,6,97,1,0,0,0,8,110,1,0,0,
        0,10,112,1,0,0,0,12,123,1,0,0,0,14,133,1,0,0,0,16,151,1,0,0,0,18,
        159,1,0,0,0,20,165,1,0,0,0,22,169,1,0,0,0,24,173,1,0,0,0,26,177,
        1,0,0,0,28,181,1,0,0,0,30,187,1,0,0,0,32,195,1,0,0,0,34,213,1,0,
        0,0,36,215,1,0,0,0,38,229,1,0,0,0,40,233,1,0,0,0,42,237,1,0,0,0,
        44,241,1,0,0,0,46,254,1,0,0,0,48,264,1,0,0,0,50,266,1,0,0,0,52,273,
        1,0,0,0,54,275,1,0,0,0,56,283,1,0,0,0,58,289,1,0,0,0,60,306,1,0,
        0,0,62,63,3,2,1,0,63,64,5,1,0,0,64,66,1,0,0,0,65,62,1,0,0,0,66,69,
        1,0,0,0,67,68,1,0,0,0,67,65,1,0,0,0,68,78,1,0,0,0,69,67,1,0,0,0,
        70,71,3,6,3,0,71,72,5,1,0,0,72,77,1,0,0,0,73,74,3,4,2,0,74,75,5,
        1,0,0,75,77,1,0,0,0,76,70,1,0,0,0,76,73,1,0,0,0,77,80,1,0,0,0,78,
        76,1,0,0,0,78,79,1,0,0,0,79,1,1,0,0,0,80,78,1,0,0,0,81,82,5,2,0,
        0,82,83,3,52,26,0,83,84,5,3,0,0,84,85,3,58,29,0,85,3,1,0,0,0,86,
        87,5,4,0,0,87,88,5,38,0,0,88,89,5,5,0,0,89,90,5,37,0,0,90,5,1,0,
        0,0,91,98,3,8,4,0,92,98,3,26,13,0,93,98,3,40,20,0,94,98,3,42,21,
        0,95,98,3,44,22,0,96,98,3,46,23,0,97,91,1,0,0,0,97,92,1,0,0,0,97,
        93,1,0,0,0,97,94,1,0,0,0,97,95,1,0,0,0,97,96,1,0,0,0,98,7,1,0,0,
        0,99,100,3,52,26,0,100,107,5,3,0,0,101,108,3,10,5,0,102,108,3,14,
        7,0,103,108,3,18,9,0,104,108,3,20,10,0,105,108,3,22,11,0,106,108,
        3,24,12,0,107,101,1,0,0,0,107,102,1,0,0,0,107,103,1,0,0,0,107,104,
        1,0,0,0,107,105,1,0,0,0,107,106,1,0,0,0,108,111,1,0,0,0,109,111,
        5,6,0,0,110,99,1,0,0,0,110,109,1,0,0,0,111,9,1,0,0,0,112,117,3,12,
        6,0,113,114,7,0,0,0,114,116,3,12,6,0,115,113,1,0,0,0,116,119,1,0,
        0,0,117,118,1,0,0,0,117,115,1,0,0,0,118,11,1,0,0,0,119,117,1,0,0,
        0,120,124,5,38,0,0,121,124,3,60,30,0,122,124,3,50,25,0,123,120,1,
        0,0,0,123,121,1,0,0,0,123,122,1,0,0,0,124,131,1,0,0,0,125,129,5,
        9,0,0,126,130,5,38,0,0,127,130,3,50,25,0,128,130,3,60,30,0,129,126,
        1,0,0,0,129,127,1,0,0,0,129,128,1,0,0,0,130,132,1,0,0,0,131,125,
        1,0,0,0,131,132,1,0,0,0,132,13,1,0,0,0,133,138,3,16,8,0,134,135,
        7,0,0,0,135,137,3,16,8,0,136,134,1,0,0,0,137,140,1,0,0,0,138,139,
        1,0,0,0,138,136,1,0,0,0,139,15,1,0,0,0,140,138,1,0,0,0,141,145,5,
        38,0,0,142,145,3,50,25,0,143,145,3,60,30,0,144,141,1,0,0,0,144,142,
        1,0,0,0,144,143,1,0,0,0,145,146,1,0,0,0,146,148,5,9,0,0,147,144,
        1,0,0,0,147,148,1,0,0,0,148,149,1,0,0,0,149,152,3,48,24,0,150,152,
        3,12,6,0,151,147,1,0,0,0,151,150,1,0,0,0,152,17,1,0,0,0,153,157,
        5,38,0,0,154,157,3,50,25,0,155,157,3,60,30,0,156,153,1,0,0,0,156,
        154,1,0,0,0,156,155,1,0,0,0,157,158,1,0,0,0,158,160,5,9,0,0,159,
        156,1,0,0,0,159,160,1,0,0,0,160,161,1,0,0,0,161,162,3,48,24,0,162,
        163,5,9,0,0,163,164,3,48,24,0,164,19,1,0,0,0,165,166,5,10,0,0,166,
        167,3,52,26,0,167,168,5,11,0,0,168,21,1,0,0,0,169,170,5,12,0,0,170,
        171,3,52,26,0,171,172,5,11,0,0,172,23,1,0,0,0,173,174,5,13,0,0,174,
        175,3,52,26,0,175,176,5,11,0,0,176,25,1,0,0,0,177,178,3,28,14,0,
        178,179,3,30,15,0,179,180,5,14,0,0,180,27,1,0,0,0,181,182,5,15,0,
        0,182,183,3,34,17,0,183,184,5,16,0,0,184,185,3,32,16,0,185,186,5,
        17,0,0,186,29,1,0,0,0,187,188,5,18,0,0,188,189,5,16,0,0,189,190,
        3,32,16,0,190,191,5,17,0,0,191,31,1,0,0,0,192,193,3,6,3,0,193,194,
        5,1,0,0,194,196,1,0,0,0,195,192,1,0,0,0,196,197,1,0,0,0,197,195,
        1,0,0,0,197,198,1,0,0,0,198,33,1,0,0,0,199,200,3,36,18,0,200,204,
        7,1,0,0,201,205,5,38,0,0,202,205,3,50,25,0,203,205,3,60,30,0,204,
        201,1,0,0,0,204,202,1,0,0,0,204,203,1,0,0,0,205,214,1,0,0,0,206,
        207,3,52,26,0,207,211,7,2,0,0,208,212,5,38,0,0,209,212,3,50,25,0,
        210,212,3,60,30,0,211,208,1,0,0,0,211,209,1,0,0,0,211,210,1,0,0,
        0,212,214,1,0,0,0,213,199,1,0,0,0,213,206,1,0,0,0,214,35,1,0,0,0,
        215,220,3,38,19,0,216,217,7,0,0,0,217,219,3,38,19,0,218,216,1,0,
        0,0,219,222,1,0,0,0,220,221,1,0,0,0,220,218,1,0,0,0,221,37,1,0,0,
        0,222,220,1,0,0,0,223,227,5,38,0,0,224,227,3,50,25,0,225,227,3,60,
        30,0,226,223,1,0,0,0,226,224,1,0,0,0,226,225,1,0,0,0,227,228,1,0,
        0,0,228,230,5,9,0,0,229,226,1,0,0,0,229,230,1,0,0,0,230,231,1,0,
        0,0,231,232,3,48,24,0,232,39,1,0,0,0,233,234,5,25,0,0,234,235,5,
        38,0,0,235,236,5,11,0,0,236,41,1,0,0,0,237,238,5,26,0,0,238,239,
        3,34,17,0,239,240,5,11,0,0,240,43,1,0,0,0,241,242,5,27,0,0,242,243,
        5,37,0,0,243,246,5,28,0,0,244,247,5,38,0,0,245,247,3,50,25,0,246,
        244,1,0,0,0,246,245,1,0,0,0,247,248,1,0,0,0,248,249,5,11,0,0,249,
        250,5,16,0,0,250,251,3,32,16,0,251,252,5,17,0,0,252,253,5,29,0,0,
        253,45,1,0,0,0,254,255,5,30,0,0,255,256,3,34,17,0,256,257,5,16,0,
        0,257,258,3,32,16,0,258,259,5,17,0,0,259,260,5,31,0,0,260,47,1,0,
        0,0,261,265,3,52,26,0,262,265,3,54,27,0,263,265,3,56,28,0,264,261,
        1,0,0,0,264,262,1,0,0,0,264,263,1,0,0,0,265,49,1,0,0,0,266,267,5,
        37,0,0,267,268,5,32,0,0,268,269,7,3,0,0,269,270,5,5,0,0,270,51,1,
        0,0,0,271,274,5,37,0,0,272,274,3,50,25,0,273,271,1,0,0,0,273,272,
        1,0,0,0,274,53,1,0,0,0,275,276,5,33,0,0,276,277,3,58,29,0,277,278,
        5,34,0,0,278,279,3,58,29,0,279,280,5,34,0,0,280,281,3,58,29,0,281,
        282,5,11,0,0,282,55,1,0,0,0,283,284,5,35,0,0,284,285,3,58,29,0,285,
        286,5,34,0,0,286,287,5,38,0,0,287,288,5,11,0,0,288,57,1,0,0,0,289,
        292,5,32,0,0,290,293,5,38,0,0,291,293,3,60,30,0,292,290,1,0,0,0,
        292,291,1,0,0,0,293,301,1,0,0,0,294,297,5,34,0,0,295,298,5,38,0,
        0,296,298,3,60,30,0,297,295,1,0,0,0,297,296,1,0,0,0,298,300,1,0,
        0,0,299,294,1,0,0,0,300,303,1,0,0,0,301,302,1,0,0,0,301,299,1,0,
        0,0,302,304,1,0,0,0,303,301,1,0,0,0,304,305,5,5,0,0,305,59,1,0,0,
        0,306,307,5,36,0,0,307,308,5,37,0,0,308,61,1,0,0,0,29,67,76,78,97,
        107,110,117,123,129,131,138,144,147,151,156,159,197,204,211,213,
        220,226,229,246,264,273,292,297,301
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
                     "'while'", "'end while'", "'['", "'gm('", "','", "'uniform('", 
                     "'_'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "IDV", "NUM", "COMM", "WS", "DIGIT" ]

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
    RULE_while = 23
    RULE_vars = 24
    RULE_idd = 25
    RULE_symvars = 26
    RULE_gm = 27
    RULE_uniform = 28
    RULE_list = 29
    RULE_par = 30

    ruleNames =  [ "progr", "data", "array", "instr", "assignment", "const", 
                   "const_term", "add", "add_term", "mul", "exp", "sin", 
                   "cos", "conditional", "ifclause", "elseclause", "block", 
                   "bexpr", "lexpr", "monom", "prune", "observe", "loop", 
                   "while", "vars", "idd", "symvars", "gm", "uniform", "list", 
                   "par" ]

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
    T__34=35
    T__35=36
    IDV=37
    NUM=38
    COMM=39
    WS=40
    DIGIT=41

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




    def progr(self):

        localctx = SOGAParser.ProgrContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_progr)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 67
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,0,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 62
                    self.data()
                    self.state = 63
                    self.match(SOGAParser.T__0) 
                self.state = 69
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,0,self._ctx)

            self.state = 78
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << SOGAParser.T__3) | (1 << SOGAParser.T__5) | (1 << SOGAParser.T__14) | (1 << SOGAParser.T__24) | (1 << SOGAParser.T__25) | (1 << SOGAParser.T__26) | (1 << SOGAParser.T__29) | (1 << SOGAParser.IDV))) != 0):
                self.state = 76
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.T__5, SOGAParser.T__14, SOGAParser.T__24, SOGAParser.T__25, SOGAParser.T__26, SOGAParser.T__29, SOGAParser.IDV]:
                    self.state = 70
                    self.instr()
                    self.state = 71
                    self.match(SOGAParser.T__0)
                    pass
                elif token in [SOGAParser.T__3]:
                    self.state = 73
                    self.array()
                    self.state = 74
                    self.match(SOGAParser.T__0)
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 80
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




    def data(self):

        localctx = SOGAParser.DataContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_data)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 81
            self.match(SOGAParser.T__1)
            self.state = 82
            self.symvars()
            self.state = 83
            self.match(SOGAParser.T__2)
            self.state = 84
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




    def array(self):

        localctx = SOGAParser.ArrayContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_array)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 86
            self.match(SOGAParser.T__3)
            self.state = 87
            self.match(SOGAParser.NUM)
            self.state = 88
            self.match(SOGAParser.T__4)
            self.state = 89
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


        def while_(self):
            return self.getTypedRuleContext(SOGAParser.WhileContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_instr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInstr" ):
                listener.enterInstr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInstr" ):
                listener.exitInstr(self)




    def instr(self):

        localctx = SOGAParser.InstrContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_instr)
        try:
            self.state = 97
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.T__5, SOGAParser.IDV]:
                self.enterOuterAlt(localctx, 1)
                self.state = 91
                self.assignment()
                pass
            elif token in [SOGAParser.T__14]:
                self.enterOuterAlt(localctx, 2)
                self.state = 92
                self.conditional()
                pass
            elif token in [SOGAParser.T__24]:
                self.enterOuterAlt(localctx, 3)
                self.state = 93
                self.prune()
                pass
            elif token in [SOGAParser.T__25]:
                self.enterOuterAlt(localctx, 4)
                self.state = 94
                self.observe()
                pass
            elif token in [SOGAParser.T__26]:
                self.enterOuterAlt(localctx, 5)
                self.state = 95
                self.loop()
                pass
            elif token in [SOGAParser.T__29]:
                self.enterOuterAlt(localctx, 6)
                self.state = 96
                self.while_()
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




    def assignment(self):

        localctx = SOGAParser.AssignmentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_assignment)
        try:
            self.state = 110
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.IDV]:
                self.enterOuterAlt(localctx, 1)
                self.state = 99
                self.symvars()
                self.state = 100
                self.match(SOGAParser.T__2)
                self.state = 107
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,4,self._ctx)
                if la_ == 1:
                    self.state = 101
                    self.const()
                    pass

                elif la_ == 2:
                    self.state = 102
                    self.add()
                    pass

                elif la_ == 3:
                    self.state = 103
                    self.mul()
                    pass

                elif la_ == 4:
                    self.state = 104
                    self.exp()
                    pass

                elif la_ == 5:
                    self.state = 105
                    self.sin()
                    pass

                elif la_ == 6:
                    self.state = 106
                    self.cos()
                    pass


                pass
            elif token in [SOGAParser.T__5]:
                self.enterOuterAlt(localctx, 2)
                self.state = 109
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




    def const(self):

        localctx = SOGAParser.ConstContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_const)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 112
            self.const_term()
            self.state = 117
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,6,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 113
                    _la = self._input.LA(1)
                    if not(_la==SOGAParser.T__6 or _la==SOGAParser.T__7):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 114
                    self.const_term() 
                self.state = 119
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




    def const_term(self):

        localctx = SOGAParser.Const_termContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_const_term)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 123
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.NUM]:
                self.state = 120
                self.match(SOGAParser.NUM)
                pass
            elif token in [SOGAParser.T__35]:
                self.state = 121
                self.par()
                pass
            elif token in [SOGAParser.IDV]:
                self.state = 122
                self.idd()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 131
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==SOGAParser.T__8:
                self.state = 125
                self.match(SOGAParser.T__8)
                self.state = 129
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 126
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 127
                    self.idd()
                    pass
                elif token in [SOGAParser.T__35]:
                    self.state = 128
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




    def add(self):

        localctx = SOGAParser.AddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_add)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 133
            self.add_term()
            self.state = 138
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,10,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 134
                    _la = self._input.LA(1)
                    if not(_la==SOGAParser.T__6 or _la==SOGAParser.T__7):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 135
                    self.add_term() 
                self.state = 140
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




    def add_term(self):

        localctx = SOGAParser.Add_termContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_add_term)
        try:
            self.state = 151
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,13,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 147
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,12,self._ctx)
                if la_ == 1:
                    self.state = 144
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [SOGAParser.NUM]:
                        self.state = 141
                        self.match(SOGAParser.NUM)
                        pass
                    elif token in [SOGAParser.IDV]:
                        self.state = 142
                        self.idd()
                        pass
                    elif token in [SOGAParser.T__35]:
                        self.state = 143
                        self.par()
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 146
                    self.match(SOGAParser.T__8)


                self.state = 149
                self.vars_()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 150
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




    def mul(self):

        localctx = SOGAParser.MulContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_mul)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 159
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,15,self._ctx)
            if la_ == 1:
                self.state = 156
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 153
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 154
                    self.idd()
                    pass
                elif token in [SOGAParser.T__35]:
                    self.state = 155
                    self.par()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 158
                self.match(SOGAParser.T__8)


            self.state = 161
            self.vars_()
            self.state = 162
            self.match(SOGAParser.T__8)
            self.state = 163
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




    def exp(self):

        localctx = SOGAParser.ExpContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_exp)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 165
            self.match(SOGAParser.T__9)
            self.state = 166
            self.symvars()
            self.state = 167
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




    def sin(self):

        localctx = SOGAParser.SinContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_sin)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 169
            self.match(SOGAParser.T__11)
            self.state = 170
            self.symvars()
            self.state = 171
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




    def cos(self):

        localctx = SOGAParser.CosContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_cos)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 173
            self.match(SOGAParser.T__12)
            self.state = 174
            self.symvars()
            self.state = 175
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




    def conditional(self):

        localctx = SOGAParser.ConditionalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_conditional)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 177
            self.ifclause()
            self.state = 178
            self.elseclause()
            self.state = 179
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




    def ifclause(self):

        localctx = SOGAParser.IfclauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_ifclause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 181
            self.match(SOGAParser.T__14)
            self.state = 182
            self.bexpr()
            self.state = 183
            self.match(SOGAParser.T__15)
            self.state = 184
            self.block()
            self.state = 185
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




    def elseclause(self):

        localctx = SOGAParser.ElseclauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_elseclause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 187
            self.match(SOGAParser.T__17)
            self.state = 188
            self.match(SOGAParser.T__15)
            self.state = 189
            self.block()
            self.state = 190
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




    def block(self):

        localctx = SOGAParser.BlockContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_block)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 195 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 192
                self.instr()
                self.state = 193
                self.match(SOGAParser.T__0)
                self.state = 197 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not ((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << SOGAParser.T__5) | (1 << SOGAParser.T__14) | (1 << SOGAParser.T__24) | (1 << SOGAParser.T__25) | (1 << SOGAParser.T__26) | (1 << SOGAParser.T__29) | (1 << SOGAParser.IDV))) != 0)):
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




    def bexpr(self):

        localctx = SOGAParser.BexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_bexpr)
        self._la = 0 # Token type
        try:
            self.state = 213
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,19,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 199
                self.lexpr()
                self.state = 200
                _la = self._input.LA(1)
                if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << SOGAParser.T__18) | (1 << SOGAParser.T__19) | (1 << SOGAParser.T__20) | (1 << SOGAParser.T__21))) != 0)):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 204
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 201
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 202
                    self.idd()
                    pass
                elif token in [SOGAParser.T__35]:
                    self.state = 203
                    self.par()
                    pass
                else:
                    raise NoViableAltException(self)

                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 206
                self.symvars()
                self.state = 207
                _la = self._input.LA(1)
                if not(_la==SOGAParser.T__22 or _la==SOGAParser.T__23):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 211
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 208
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 209
                    self.idd()
                    pass
                elif token in [SOGAParser.T__35]:
                    self.state = 210
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




    def lexpr(self):

        localctx = SOGAParser.LexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_lexpr)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 215
            self.monom()
            self.state = 220
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,20,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 216
                    _la = self._input.LA(1)
                    if not(_la==SOGAParser.T__6 or _la==SOGAParser.T__7):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 217
                    self.monom() 
                self.state = 222
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




    def monom(self):

        localctx = SOGAParser.MonomContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_monom)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 229
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,22,self._ctx)
            if la_ == 1:
                self.state = 226
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 223
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 224
                    self.idd()
                    pass
                elif token in [SOGAParser.T__35]:
                    self.state = 225
                    self.par()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 228
                self.match(SOGAParser.T__8)


            self.state = 231
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




    def prune(self):

        localctx = SOGAParser.PruneContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_prune)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 233
            self.match(SOGAParser.T__24)
            self.state = 234
            self.match(SOGAParser.NUM)
            self.state = 235
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




    def observe(self):

        localctx = SOGAParser.ObserveContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_observe)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 237
            self.match(SOGAParser.T__25)
            self.state = 238
            self.bexpr()
            self.state = 239
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




    def loop(self):

        localctx = SOGAParser.LoopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_loop)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 241
            self.match(SOGAParser.T__26)
            self.state = 242
            self.match(SOGAParser.IDV)
            self.state = 243
            self.match(SOGAParser.T__27)
            self.state = 246
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.NUM]:
                self.state = 244
                self.match(SOGAParser.NUM)
                pass
            elif token in [SOGAParser.IDV]:
                self.state = 245
                self.idd()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 248
            self.match(SOGAParser.T__10)
            self.state = 249
            self.match(SOGAParser.T__15)
            self.state = 250
            self.block()
            self.state = 251
            self.match(SOGAParser.T__16)
            self.state = 252
            self.match(SOGAParser.T__28)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class WhileContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def bexpr(self):
            return self.getTypedRuleContext(SOGAParser.BexprContext,0)


        def block(self):
            return self.getTypedRuleContext(SOGAParser.BlockContext,0)


        def getRuleIndex(self):
            return SOGAParser.RULE_while

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterWhile" ):
                listener.enterWhile(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitWhile" ):
                listener.exitWhile(self)




    def while_(self):

        localctx = SOGAParser.WhileContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_while)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 254
            self.match(SOGAParser.T__29)
            self.state = 255
            self.bexpr()
            self.state = 256
            self.match(SOGAParser.T__15)
            self.state = 257
            self.block()
            self.state = 258
            self.match(SOGAParser.T__16)
            self.state = 259
            self.match(SOGAParser.T__30)
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




    def vars_(self):

        localctx = SOGAParser.VarsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_vars)
        try:
            self.state = 264
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.IDV]:
                self.enterOuterAlt(localctx, 1)
                self.state = 261
                self.symvars()
                pass
            elif token in [SOGAParser.T__32]:
                self.enterOuterAlt(localctx, 2)
                self.state = 262
                self.gm()
                pass
            elif token in [SOGAParser.T__34]:
                self.enterOuterAlt(localctx, 3)
                self.state = 263
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




    def idd(self):

        localctx = SOGAParser.IddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_idd)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 266
            self.match(SOGAParser.IDV)
            self.state = 267
            self.match(SOGAParser.T__31)
            self.state = 268
            _la = self._input.LA(1)
            if not(_la==SOGAParser.IDV or _la==SOGAParser.NUM):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 269
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




    def symvars(self):

        localctx = SOGAParser.SymvarsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_symvars)
        try:
            self.state = 273
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,25,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 271
                self.match(SOGAParser.IDV)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 272
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




    def gm(self):

        localctx = SOGAParser.GmContext(self, self._ctx, self.state)
        self.enterRule(localctx, 54, self.RULE_gm)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 275
            self.match(SOGAParser.T__32)
            self.state = 276
            self.list_()
            self.state = 277
            self.match(SOGAParser.T__33)
            self.state = 278
            self.list_()
            self.state = 279
            self.match(SOGAParser.T__33)
            self.state = 280
            self.list_()
            self.state = 281
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

        def getText(self):
            """ converts string "uniform([a,b], K)" in "gm(pi, mu, sigma)" where gm is a Gaussian Mix with K component approximating the uniform"""
            a = float(self.list_().NUM()[0].getText())
            b = float(self.list_().NUM()[1].getText())
            N = int(self.NUM().getText())
            pi = [round(1.0/N,4)]*N
            mu = [round(a+i*(b-a)/N+((b-a)/(2*N)),4) for i in range(N)]
            #sigma = list([float(round((b-a)/(np.sqrt(12)*N/np.log(N)),4))]*N)
            sigma = list([float(round((b-a)/(np.sqrt(12)*N),4))]*N)
            return 'gm('+str(pi)+','+str(mu)+','+str(sigma)+')'




    def uniform(self):

        localctx = SOGAParser.UniformContext(self, self._ctx, self.state)
        self.enterRule(localctx, 56, self.RULE_uniform)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 283
            self.match(SOGAParser.T__34)
            self.state = 284
            self.list_()
            self.state = 285
            self.match(SOGAParser.T__33)
            self.state = 286
            self.match(SOGAParser.NUM)
            self.state = 287
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




    def list_(self):

        localctx = SOGAParser.ListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 58, self.RULE_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 289
            self.match(SOGAParser.T__31)
            self.state = 292
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.NUM]:
                self.state = 290
                self.match(SOGAParser.NUM)
                pass
            elif token in [SOGAParser.T__35]:
                self.state = 291
                self.par()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 301
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,28,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 294
                    self.match(SOGAParser.T__33)
                    self.state = 297
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [SOGAParser.NUM]:
                        self.state = 295
                        self.match(SOGAParser.NUM)
                        pass
                    elif token in [SOGAParser.T__35]:
                        self.state = 296
                        self.par()
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 303
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,28,self._ctx)

            self.state = 304
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




    def par(self):

        localctx = SOGAParser.ParContext(self, self._ctx, self.state)
        self.enterRule(localctx, 60, self.RULE_par)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 306
            self.match(SOGAParser.T__35)
            self.state = 307
            self.match(SOGAParser.IDV)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





