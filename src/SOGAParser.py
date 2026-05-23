# Generated from SOGA.g4 by ANTLR 4.10
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO
import numpy as np  # required by custom UniformContext.getText (preserved across regen)

def serializedATN():
    return [
        4,1,39,316,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        2,27,7,27,2,28,7,28,2,29,7,29,1,0,1,0,1,0,5,0,64,8,0,10,0,12,0,67,
        9,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,5,0,78,8,0,10,0,12,0,81,
        9,0,1,1,1,1,1,1,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1,3,
        1,3,1,3,1,3,1,3,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,5,1,5,1,5,
        1,5,5,5,115,8,5,10,5,12,5,118,9,5,1,5,1,5,1,6,1,6,1,6,1,6,1,6,3,
        6,127,8,6,1,7,1,7,1,7,1,7,1,7,3,7,134,8,7,1,7,3,7,137,8,7,1,8,1,
        8,1,8,5,8,142,8,8,10,8,12,8,145,9,8,1,9,1,9,3,9,149,8,9,1,9,1,9,
        1,9,3,9,154,8,9,3,9,156,8,9,1,10,1,10,1,10,5,10,161,8,10,10,10,12,
        10,164,9,10,1,11,1,11,3,11,168,8,11,1,11,3,11,171,8,11,1,11,1,11,
        3,11,175,8,11,1,12,1,12,3,12,179,8,12,1,12,3,12,182,8,12,1,12,1,
        12,1,12,1,12,1,13,1,13,1,13,1,13,1,14,1,14,1,14,1,14,1,14,1,14,1,
        15,1,15,1,15,1,15,1,15,1,16,1,16,1,16,4,16,206,8,16,11,16,12,16,
        207,1,17,1,17,1,17,1,17,3,17,214,8,17,1,17,1,17,1,17,1,17,3,17,220,
        8,17,3,17,222,8,17,1,18,1,18,1,18,5,18,227,8,18,10,18,12,18,230,
        9,18,1,19,1,19,3,19,234,8,19,1,19,3,19,237,8,19,1,19,1,19,1,20,1,
        20,1,20,1,20,1,21,1,21,1,21,1,21,1,22,1,22,1,22,1,22,1,22,3,22,254,
        8,22,1,22,1,22,1,22,1,22,1,22,1,22,1,23,1,23,1,23,3,23,265,8,23,
        1,23,1,23,1,23,1,23,1,23,1,23,1,23,3,23,274,8,23,1,24,1,24,1,24,
        1,24,3,24,280,8,24,1,25,1,25,1,25,1,25,1,25,1,26,1,26,3,26,289,8,
        26,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,28,1,28,1,28,1,28,1,
        28,1,28,1,29,1,29,1,29,1,29,5,29,309,8,29,10,29,12,29,312,9,29,1,
        29,1,29,1,29,5,65,143,162,228,310,0,30,0,2,4,6,8,10,12,14,16,18,
        20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,0,4,
        1,0,11,12,1,0,19,22,1,0,23,24,1,0,35,36,323,0,65,1,0,0,0,2,82,1,
        0,0,0,4,87,1,0,0,0,6,92,1,0,0,0,8,101,1,0,0,0,10,110,1,0,0,0,12,
        126,1,0,0,0,14,136,1,0,0,0,16,138,1,0,0,0,18,148,1,0,0,0,20,157,
        1,0,0,0,22,174,1,0,0,0,24,181,1,0,0,0,26,187,1,0,0,0,28,191,1,0,
        0,0,30,197,1,0,0,0,32,205,1,0,0,0,34,221,1,0,0,0,36,223,1,0,0,0,
        38,236,1,0,0,0,40,240,1,0,0,0,42,244,1,0,0,0,44,248,1,0,0,0,46,273,
        1,0,0,0,48,279,1,0,0,0,50,281,1,0,0,0,52,288,1,0,0,0,54,290,1,0,
        0,0,56,298,1,0,0,0,58,304,1,0,0,0,60,61,3,2,1,0,61,62,5,1,0,0,62,
        64,1,0,0,0,63,60,1,0,0,0,64,67,1,0,0,0,65,66,1,0,0,0,65,63,1,0,0,
        0,66,79,1,0,0,0,67,65,1,0,0,0,68,69,3,12,6,0,69,70,5,1,0,0,70,78,
        1,0,0,0,71,72,3,4,2,0,72,73,5,1,0,0,73,78,1,0,0,0,74,75,3,6,3,0,
        75,76,5,1,0,0,76,78,1,0,0,0,77,68,1,0,0,0,77,71,1,0,0,0,77,74,1,
        0,0,0,78,81,1,0,0,0,79,77,1,0,0,0,79,80,1,0,0,0,80,1,1,0,0,0,81,
        79,1,0,0,0,82,83,5,2,0,0,83,84,3,52,26,0,84,85,5,3,0,0,85,86,3,58,
        29,0,86,3,1,0,0,0,87,88,5,4,0,0,88,89,5,36,0,0,89,90,5,5,0,0,90,
        91,5,35,0,0,91,5,1,0,0,0,92,93,5,33,0,0,93,94,5,6,0,0,94,95,5,36,
        0,0,95,96,5,5,0,0,96,97,5,6,0,0,97,98,5,36,0,0,98,99,5,5,0,0,99,
        100,5,35,0,0,100,7,1,0,0,0,101,102,5,34,0,0,102,103,5,7,0,0,103,
        104,3,10,5,0,104,105,5,8,0,0,105,106,3,10,5,0,106,107,5,8,0,0,107,
        108,3,10,5,0,108,109,5,9,0,0,109,9,1,0,0,0,110,111,5,6,0,0,111,116,
        3,58,29,0,112,113,5,8,0,0,113,115,3,58,29,0,114,112,1,0,0,0,115,
        118,1,0,0,0,116,114,1,0,0,0,116,117,1,0,0,0,117,119,1,0,0,0,118,
        116,1,0,0,0,119,120,5,5,0,0,120,11,1,0,0,0,121,127,3,14,7,0,122,
        127,3,26,13,0,123,127,3,40,20,0,124,127,3,42,21,0,125,127,3,44,22,
        0,126,121,1,0,0,0,126,122,1,0,0,0,126,123,1,0,0,0,126,124,1,0,0,
        0,126,125,1,0,0,0,127,13,1,0,0,0,128,129,3,52,26,0,129,133,5,3,0,
        0,130,134,3,16,8,0,131,134,3,20,10,0,132,134,3,24,12,0,133,130,1,
        0,0,0,133,131,1,0,0,0,133,132,1,0,0,0,134,137,1,0,0,0,135,137,5,
        10,0,0,136,128,1,0,0,0,136,135,1,0,0,0,137,15,1,0,0,0,138,143,3,
        18,9,0,139,140,7,0,0,0,140,142,3,18,9,0,141,139,1,0,0,0,142,145,
        1,0,0,0,143,144,1,0,0,0,143,141,1,0,0,0,144,17,1,0,0,0,145,143,1,
        0,0,0,146,149,5,36,0,0,147,149,3,50,25,0,148,146,1,0,0,0,148,147,
        1,0,0,0,149,155,1,0,0,0,150,153,5,13,0,0,151,154,5,36,0,0,152,154,
        3,50,25,0,153,151,1,0,0,0,153,152,1,0,0,0,154,156,1,0,0,0,155,150,
        1,0,0,0,155,156,1,0,0,0,156,19,1,0,0,0,157,162,3,22,11,0,158,159,
        7,0,0,0,159,161,3,22,11,0,160,158,1,0,0,0,161,164,1,0,0,0,162,163,
        1,0,0,0,162,160,1,0,0,0,163,21,1,0,0,0,164,162,1,0,0,0,165,168,5,
        36,0,0,166,168,3,50,25,0,167,165,1,0,0,0,167,166,1,0,0,0,168,169,
        1,0,0,0,169,171,5,13,0,0,170,167,1,0,0,0,170,171,1,0,0,0,171,172,
        1,0,0,0,172,175,3,48,24,0,173,175,3,18,9,0,174,170,1,0,0,0,174,173,
        1,0,0,0,175,23,1,0,0,0,176,179,5,36,0,0,177,179,3,50,25,0,178,176,
        1,0,0,0,178,177,1,0,0,0,179,180,1,0,0,0,180,182,5,13,0,0,181,178,
        1,0,0,0,181,182,1,0,0,0,182,183,1,0,0,0,183,184,3,48,24,0,184,185,
        5,13,0,0,185,186,3,48,24,0,186,25,1,0,0,0,187,188,3,28,14,0,188,
        189,3,30,15,0,189,190,5,14,0,0,190,27,1,0,0,0,191,192,5,15,0,0,192,
        193,3,34,17,0,193,194,5,16,0,0,194,195,3,32,16,0,195,196,5,17,0,
        0,196,29,1,0,0,0,197,198,5,18,0,0,198,199,5,16,0,0,199,200,3,32,
        16,0,200,201,5,17,0,0,201,31,1,0,0,0,202,203,3,12,6,0,203,204,5,
        1,0,0,204,206,1,0,0,0,205,202,1,0,0,0,206,207,1,0,0,0,207,205,1,
        0,0,0,207,208,1,0,0,0,208,33,1,0,0,0,209,210,3,36,18,0,210,213,7,
        1,0,0,211,214,5,36,0,0,212,214,3,50,25,0,213,211,1,0,0,0,213,212,
        1,0,0,0,214,222,1,0,0,0,215,216,3,52,26,0,216,219,7,2,0,0,217,220,
        5,36,0,0,218,220,3,50,25,0,219,217,1,0,0,0,219,218,1,0,0,0,220,222,
        1,0,0,0,221,209,1,0,0,0,221,215,1,0,0,0,222,35,1,0,0,0,223,228,3,
        38,19,0,224,225,7,0,0,0,225,227,3,38,19,0,226,224,1,0,0,0,227,230,
        1,0,0,0,228,229,1,0,0,0,228,226,1,0,0,0,229,37,1,0,0,0,230,228,1,
        0,0,0,231,234,5,36,0,0,232,234,3,50,25,0,233,231,1,0,0,0,233,232,
        1,0,0,0,234,235,1,0,0,0,235,237,5,13,0,0,236,233,1,0,0,0,236,237,
        1,0,0,0,237,238,1,0,0,0,238,239,3,48,24,0,239,39,1,0,0,0,240,241,
        5,25,0,0,241,242,5,36,0,0,242,243,5,9,0,0,243,41,1,0,0,0,244,245,
        5,26,0,0,245,246,3,34,17,0,246,247,5,9,0,0,247,43,1,0,0,0,248,249,
        5,27,0,0,249,250,5,35,0,0,250,253,5,28,0,0,251,254,5,36,0,0,252,
        254,3,50,25,0,253,251,1,0,0,0,253,252,1,0,0,0,254,255,1,0,0,0,255,
        256,5,9,0,0,256,257,5,16,0,0,257,258,3,32,16,0,258,259,5,17,0,0,
        259,260,5,29,0,0,260,45,1,0,0,0,261,274,3,36,18,0,262,263,5,36,0,
        0,263,265,5,13,0,0,264,262,1,0,0,0,264,265,1,0,0,0,265,266,1,0,0,
        0,266,267,3,48,24,0,267,268,5,13,0,0,268,269,3,48,24,0,269,274,1,
        0,0,0,270,271,3,48,24,0,271,272,5,30,0,0,272,274,1,0,0,0,273,261,
        1,0,0,0,273,264,1,0,0,0,273,270,1,0,0,0,274,47,1,0,0,0,275,280,3,
        52,26,0,276,280,3,54,27,0,277,280,3,56,28,0,278,280,3,8,4,0,279,
        275,1,0,0,0,279,276,1,0,0,0,279,277,1,0,0,0,279,278,1,0,0,0,280,
        49,1,0,0,0,281,282,5,35,0,0,282,283,5,6,0,0,283,284,7,3,0,0,284,
        285,5,5,0,0,285,51,1,0,0,0,286,289,5,35,0,0,287,289,3,50,25,0,288,
        286,1,0,0,0,288,287,1,0,0,0,289,53,1,0,0,0,290,291,5,31,0,0,291,
        292,3,58,29,0,292,293,5,8,0,0,293,294,3,58,29,0,294,295,5,8,0,0,
        295,296,3,58,29,0,296,297,5,9,0,0,297,55,1,0,0,0,298,299,5,32,0,
        0,299,300,3,58,29,0,300,301,5,8,0,0,301,302,5,36,0,0,302,303,5,9,
        0,0,303,57,1,0,0,0,304,305,5,6,0,0,305,310,5,36,0,0,306,307,5,8,
        0,0,307,309,5,36,0,0,308,306,1,0,0,0,309,312,1,0,0,0,310,311,1,0,
        0,0,310,308,1,0,0,0,311,313,1,0,0,0,312,310,1,0,0,0,313,314,5,5,
        0,0,314,59,1,0,0,0,30,65,77,79,116,126,133,136,143,148,153,155,162,
        167,170,174,178,181,207,213,219,221,228,233,236,253,264,273,279,
        288,310
    ]

class SOGAParser ( Parser ):

    grammarFileName = "SOGA.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "';'", "'data'", "'='", "'array['", "']'", 
                     "'['", "'('", "','", "')'", "'skip'", "'+'", "'-'", 
                     "'*'", "'end if'", "'if'", "'{'", "'}'", "'else'", 
                     "'<'", "'<='", "'>='", "'>'", "'=='", "'!='", "'prune('", 
                     "'observe('", "'for'", "'in range('", "'end for'", 
                     "'^2'", "'gm('", "'uniform('", "'matrix'", "'matrix_gm'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "MATRIX", "MATRIX_GM", "IDV", "NUM", 
                      "COMM", "WS", "DIGIT" ]

    RULE_progr = 0
    RULE_data = 1
    RULE_array = 2
    RULE_matrix_decl = 3
    RULE_matrix_gm = 4
    RULE_mlist = 5
    RULE_instr = 6
    RULE_assignment = 7
    RULE_const = 8
    RULE_const_term = 9
    RULE_add = 10
    RULE_add_term = 11
    RULE_mul = 12
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
    RULE_expr = 23
    RULE_vars = 24
    RULE_idd = 25
    RULE_symvars = 26
    RULE_gm = 27
    RULE_uniform = 28
    RULE_list = 29

    ruleNames =  [ "progr", "data", "array", "matrix_decl", "matrix_gm", 
                   "mlist", "instr", "assignment", "const", "const_term", 
                   "add", "add_term", "mul", "conditional", "ifclause", 
                   "elseclause", "block", "bexpr", "lexpr", "monom", "prune", 
                   "observe", "loop", "expr", "vars", "idd", "symvars", 
                   "gm", "uniform", "list" ]

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
    MATRIX=33
    MATRIX_GM=34
    IDV=35
    NUM=36
    COMM=37
    WS=38
    DIGIT=39

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10")
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


        def matrix_decl(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.Matrix_declContext)
            else:
                return self.getTypedRuleContext(SOGAParser.Matrix_declContext,i)


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

            self.state = 79
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << SOGAParser.T__3) | (1 << SOGAParser.T__9) | (1 << SOGAParser.T__14) | (1 << SOGAParser.T__24) | (1 << SOGAParser.T__25) | (1 << SOGAParser.T__26) | (1 << SOGAParser.MATRIX) | (1 << SOGAParser.IDV))) != 0):
                self.state = 77
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.T__9, SOGAParser.T__14, SOGAParser.T__24, SOGAParser.T__25, SOGAParser.T__26, SOGAParser.IDV]:
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
                elif token in [SOGAParser.MATRIX]:
                    self.state = 74
                    self.matrix_decl()
                    self.state = 75
                    self.match(SOGAParser.T__0)
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 81
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
            self.state = 82
            self.match(SOGAParser.T__1)
            self.state = 83
            self.symvars()
            self.state = 84
            self.match(SOGAParser.T__2)
            self.state = 85
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
            self.state = 87
            self.match(SOGAParser.T__3)
            self.state = 88
            self.match(SOGAParser.NUM)
            self.state = 89
            self.match(SOGAParser.T__4)
            self.state = 90
            self.match(SOGAParser.IDV)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Matrix_declContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def MATRIX(self):
            return self.getToken(SOGAParser.MATRIX, 0)

        def NUM(self, i:int=None):
            if i is None:
                return self.getTokens(SOGAParser.NUM)
            else:
                return self.getToken(SOGAParser.NUM, i)

        def IDV(self):
            return self.getToken(SOGAParser.IDV, 0)

        def getRuleIndex(self):
            return SOGAParser.RULE_matrix_decl

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatrix_decl" ):
                listener.enterMatrix_decl(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatrix_decl" ):
                listener.exitMatrix_decl(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMatrix_decl" ):
                return visitor.visitMatrix_decl(self)
            else:
                return visitor.visitChildren(self)




    def matrix_decl(self):

        localctx = SOGAParser.Matrix_declContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_matrix_decl)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 92
            self.match(SOGAParser.MATRIX)
            self.state = 93
            self.match(SOGAParser.T__5)
            self.state = 94
            self.match(SOGAParser.NUM)
            self.state = 95
            self.match(SOGAParser.T__4)
            self.state = 96
            self.match(SOGAParser.T__5)
            self.state = 97
            self.match(SOGAParser.NUM)
            self.state = 98
            self.match(SOGAParser.T__4)
            self.state = 99
            self.match(SOGAParser.IDV)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Matrix_gmContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def MATRIX_GM(self):
            return self.getToken(SOGAParser.MATRIX_GM, 0)

        def mlist(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.MlistContext)
            else:
                return self.getTypedRuleContext(SOGAParser.MlistContext,i)


        def getRuleIndex(self):
            return SOGAParser.RULE_matrix_gm

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatrix_gm" ):
                listener.enterMatrix_gm(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatrix_gm" ):
                listener.exitMatrix_gm(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMatrix_gm" ):
                return visitor.visitMatrix_gm(self)
            else:
                return visitor.visitChildren(self)




    def matrix_gm(self):

        localctx = SOGAParser.Matrix_gmContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_matrix_gm)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 101
            self.match(SOGAParser.MATRIX_GM)
            self.state = 102
            self.match(SOGAParser.T__6)
            self.state = 103
            self.mlist()
            self.state = 104
            self.match(SOGAParser.T__7)
            self.state = 105
            self.mlist()
            self.state = 106
            self.match(SOGAParser.T__7)
            self.state = 107
            self.mlist()
            self.state = 108
            self.match(SOGAParser.T__8)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MlistContext(ParserRuleContext):
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
            return SOGAParser.RULE_mlist

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMlist" ):
                listener.enterMlist(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMlist" ):
                listener.exitMlist(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMlist" ):
                return visitor.visitMlist(self)
            else:
                return visitor.visitChildren(self)




    def mlist(self):

        localctx = SOGAParser.MlistContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_mlist)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 110
            self.match(SOGAParser.T__5)
            self.state = 111
            self.list_()
            self.state = 116
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==SOGAParser.T__7:
                self.state = 112
                self.match(SOGAParser.T__7)
                self.state = 113
                self.list_()
                self.state = 118
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 119
            self.match(SOGAParser.T__4)
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
        self.enterRule(localctx, 12, self.RULE_instr)
        try:
            self.state = 126
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.T__9, SOGAParser.IDV]:
                self.enterOuterAlt(localctx, 1)
                self.state = 121
                self.assignment()
                pass
            elif token in [SOGAParser.T__14]:
                self.enterOuterAlt(localctx, 2)
                self.state = 122
                self.conditional()
                pass
            elif token in [SOGAParser.T__24]:
                self.enterOuterAlt(localctx, 3)
                self.state = 123
                self.prune()
                pass
            elif token in [SOGAParser.T__25]:
                self.enterOuterAlt(localctx, 4)
                self.state = 124
                self.observe()
                pass
            elif token in [SOGAParser.T__26]:
                self.enterOuterAlt(localctx, 5)
                self.state = 125
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
        self.enterRule(localctx, 14, self.RULE_assignment)
        try:
            self.state = 136
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.IDV]:
                self.enterOuterAlt(localctx, 1)
                self.state = 128
                self.symvars()
                self.state = 129
                self.match(SOGAParser.T__2)
                self.state = 133
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
                if la_ == 1:
                    self.state = 130
                    self.const()
                    pass

                elif la_ == 2:
                    self.state = 131
                    self.add()
                    pass

                elif la_ == 3:
                    self.state = 132
                    self.mul()
                    pass


                pass
            elif token in [SOGAParser.T__9]:
                self.enterOuterAlt(localctx, 2)
                self.state = 135
                self.match(SOGAParser.T__9)
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
        self.enterRule(localctx, 16, self.RULE_const)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 138
            self.const_term()
            self.state = 143
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,7,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 139
                    _la = self._input.LA(1)
                    if not(_la==SOGAParser.T__10 or _la==SOGAParser.T__11):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 140
                    self.const_term() 
                self.state = 145
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,7,self._ctx)

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
        self.enterRule(localctx, 18, self.RULE_const_term)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 148
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.NUM]:
                self.state = 146
                self.match(SOGAParser.NUM)
                pass
            elif token in [SOGAParser.IDV]:
                self.state = 147
                self.idd()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 155
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==SOGAParser.T__12:
                self.state = 150
                self.match(SOGAParser.T__12)
                self.state = 153
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 151
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 152
                    self.idd()
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
        self.enterRule(localctx, 20, self.RULE_add)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 157
            self.add_term()
            self.state = 162
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,11,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 158
                    _la = self._input.LA(1)
                    if not(_la==SOGAParser.T__10 or _la==SOGAParser.T__11):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 159
                    self.add_term() 
                self.state = 164
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,11,self._ctx)

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
        self.enterRule(localctx, 22, self.RULE_add_term)
        try:
            self.state = 174
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,14,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 170
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,13,self._ctx)
                if la_ == 1:
                    self.state = 167
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [SOGAParser.NUM]:
                        self.state = 165
                        self.match(SOGAParser.NUM)
                        pass
                    elif token in [SOGAParser.IDV]:
                        self.state = 166
                        self.idd()
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 169
                    self.match(SOGAParser.T__12)


                self.state = 172
                self.vars_()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 173
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
        self.enterRule(localctx, 24, self.RULE_mul)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 181
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,16,self._ctx)
            if la_ == 1:
                self.state = 178
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 176
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 177
                    self.idd()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 180
                self.match(SOGAParser.T__12)


            self.state = 183
            self.vars_()
            self.state = 184
            self.match(SOGAParser.T__12)
            self.state = 185
            self.vars_()
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
            self.state = 187
            self.ifclause()
            self.state = 188
            self.elseclause()
            self.state = 189
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
            self.state = 191
            self.match(SOGAParser.T__14)
            self.state = 192
            self.bexpr()
            self.state = 193
            self.match(SOGAParser.T__15)
            self.state = 194
            self.block()
            self.state = 195
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
            self.state = 197
            self.match(SOGAParser.T__17)
            self.state = 198
            self.match(SOGAParser.T__15)
            self.state = 199
            self.block()
            self.state = 200
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
            self.state = 205 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 202
                self.instr()
                self.state = 203
                self.match(SOGAParser.T__0)
                self.state = 207 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not ((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << SOGAParser.T__9) | (1 << SOGAParser.T__14) | (1 << SOGAParser.T__24) | (1 << SOGAParser.T__25) | (1 << SOGAParser.T__26) | (1 << SOGAParser.IDV))) != 0)):
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
            self.state = 221
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,20,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 209
                self.lexpr()
                self.state = 210
                _la = self._input.LA(1)
                if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << SOGAParser.T__18) | (1 << SOGAParser.T__19) | (1 << SOGAParser.T__20) | (1 << SOGAParser.T__21))) != 0)):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 213
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 211
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 212
                    self.idd()
                    pass
                else:
                    raise NoViableAltException(self)

                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 215
                self.symvars()
                self.state = 216
                _la = self._input.LA(1)
                if not(_la==SOGAParser.T__22 or _la==SOGAParser.T__23):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 219
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 217
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 218
                    self.idd()
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
            self.state = 223
            self.monom()
            self.state = 228
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,21,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 224
                    _la = self._input.LA(1)
                    if not(_la==SOGAParser.T__10 or _la==SOGAParser.T__11):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 225
                    self.monom() 
                self.state = 230
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,21,self._ctx)

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
            self.state = 236
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,23,self._ctx)
            if la_ == 1:
                self.state = 233
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [SOGAParser.NUM]:
                    self.state = 231
                    self.match(SOGAParser.NUM)
                    pass
                elif token in [SOGAParser.IDV]:
                    self.state = 232
                    self.idd()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 235
                self.match(SOGAParser.T__12)


            self.state = 238
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
            self.state = 240
            self.match(SOGAParser.T__24)
            self.state = 241
            self.match(SOGAParser.NUM)
            self.state = 242
            self.match(SOGAParser.T__8)
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
            self.state = 244
            self.match(SOGAParser.T__25)
            self.state = 245
            self.bexpr()
            self.state = 246
            self.match(SOGAParser.T__8)
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
            self.state = 248
            self.match(SOGAParser.T__26)
            self.state = 249
            self.match(SOGAParser.IDV)
            self.state = 250
            self.match(SOGAParser.T__27)
            self.state = 253
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.NUM]:
                self.state = 251
                self.match(SOGAParser.NUM)
                pass
            elif token in [SOGAParser.IDV]:
                self.state = 252
                self.idd()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 255
            self.match(SOGAParser.T__8)
            self.state = 256
            self.match(SOGAParser.T__15)
            self.state = 257
            self.block()
            self.state = 258
            self.match(SOGAParser.T__16)
            self.state = 259
            self.match(SOGAParser.T__28)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def lexpr(self):
            return self.getTypedRuleContext(SOGAParser.LexprContext,0)


        def vars_(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SOGAParser.VarsContext)
            else:
                return self.getTypedRuleContext(SOGAParser.VarsContext,i)


        def NUM(self):
            return self.getToken(SOGAParser.NUM, 0)

        def getRuleIndex(self):
            return SOGAParser.RULE_expr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExpr" ):
                listener.enterExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExpr" ):
                listener.exitExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExpr" ):
                return visitor.visitExpr(self)
            else:
                return visitor.visitChildren(self)




    def expr(self):

        localctx = SOGAParser.ExprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_expr)
        self._la = 0 # Token type
        try:
            self.state = 273
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,26,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 261
                self.lexpr()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 264
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==SOGAParser.NUM:
                    self.state = 262
                    self.match(SOGAParser.NUM)
                    self.state = 263
                    self.match(SOGAParser.T__12)


                self.state = 266
                self.vars_()
                self.state = 267
                self.match(SOGAParser.T__12)
                self.state = 268
                self.vars_()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 270
                self.vars_()
                self.state = 271
                self.match(SOGAParser.T__29)
                pass


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


        def matrix_gm(self):
            return self.getTypedRuleContext(SOGAParser.Matrix_gmContext,0)


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
        self.enterRule(localctx, 48, self.RULE_vars)
        try:
            self.state = 279
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SOGAParser.IDV]:
                self.enterOuterAlt(localctx, 1)
                self.state = 275
                self.symvars()
                pass
            elif token in [SOGAParser.T__30]:
                self.enterOuterAlt(localctx, 2)
                self.state = 276
                self.gm()
                pass
            elif token in [SOGAParser.T__31]:
                self.enterOuterAlt(localctx, 3)
                self.state = 277
                self.uniform()
                pass
            elif token in [SOGAParser.MATRIX_GM]:
                self.enterOuterAlt(localctx, 4)
                self.state = 278
                self.matrix_gm()
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
        self.enterRule(localctx, 50, self.RULE_idd)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 281
            self.match(SOGAParser.IDV)
            self.state = 282
            self.match(SOGAParser.T__5)
            self.state = 283
            _la = self._input.LA(1)
            if not(_la==SOGAParser.IDV or _la==SOGAParser.NUM):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 284
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
        self.enterRule(localctx, 52, self.RULE_symvars)
        try:
            self.state = 288
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,28,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 286
                self.match(SOGAParser.IDV)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 287
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
        self.enterRule(localctx, 54, self.RULE_gm)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 290
            self.match(SOGAParser.T__30)
            self.state = 291
            self.list_()
            self.state = 292
            self.match(SOGAParser.T__7)
            self.state = 293
            self.list_()
            self.state = 294
            self.match(SOGAParser.T__7)
            self.state = 295
            self.list_()
            self.state = 296
            self.match(SOGAParser.T__8)
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

        # --- BEGIN custom methods (preserved across ANTLR regenerations) ---
        def getText(self):
            """ converts string "uniform([a,b], K)" in "gm(pi, mu, sigma)" where gm is a Gaussian Mix with K component approximating the uniform"""
            a = float(self.list_().NUM()[0].getText())
            b = float(self.list_().NUM()[1].getText())
            N = int(self.NUM().getText())
            pi = [round(1.0/N,4)]*N
            mu = [round(a+i*(b-a)/N+((b-a)/(2*N)),4) for i in range(N)]
            sigma = list([round((b-a)/(np.sqrt(12)*N),4)]*N)
            print('gm('+str(pi)+','+str(mu)+','+str(sigma)+')')
            return 'gm('+str(pi)+','+str(mu)+','+str(sigma)+')'
        # --- END custom methods ---


    def uniform(self):

        localctx = SOGAParser.UniformContext(self, self._ctx, self.state)
        self.enterRule(localctx, 56, self.RULE_uniform)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 298
            self.match(SOGAParser.T__31)
            self.state = 299
            self.list_()
            self.state = 300
            self.match(SOGAParser.T__7)
            self.state = 301
            self.match(SOGAParser.NUM)
            self.state = 302
            self.match(SOGAParser.T__8)
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
        self.enterRule(localctx, 58, self.RULE_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 304
            self.match(SOGAParser.T__5)
            self.state = 305
            self.match(SOGAParser.NUM)
            self.state = 310
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,29,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 306
                    self.match(SOGAParser.T__7)
                    self.state = 307
                    self.match(SOGAParser.NUM) 
                self.state = 312
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,29,self._ctx)

            self.state = 313
            self.match(SOGAParser.T__4)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





