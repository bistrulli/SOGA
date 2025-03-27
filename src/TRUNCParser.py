# Generated from TRUNC.g4 by ANTLR 4.10.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO
import torch

def serializedATN():
    return [
        4,1,23,135,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,1,0,1,0,1,0,1,0,3,0,39,8,0,1,1,1,1,
        1,1,1,1,1,2,1,2,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,4,1,4,1,4,1,4,
        1,4,1,4,1,4,1,4,1,5,1,5,1,5,1,5,1,6,1,6,1,7,1,7,1,7,3,7,72,8,7,1,
        7,1,7,5,7,76,8,7,10,7,12,7,79,9,7,1,8,1,8,1,8,3,8,84,8,8,1,8,1,8,
        1,9,1,9,1,9,3,9,91,8,9,1,10,1,10,1,10,3,10,96,8,10,1,11,1,11,1,11,
        1,11,1,11,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,13,1,13,1,13,
        3,13,114,8,13,1,13,1,13,1,13,3,13,119,8,13,5,13,121,8,13,10,13,12,
        13,124,9,13,1,13,1,13,1,14,1,14,1,15,1,15,1,16,1,16,1,16,1,16,2,
        77,122,0,17,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,0,3,1,
        0,1,4,1,0,7,8,1,0,18,19,130,0,38,1,0,0,0,2,40,1,0,0,0,4,44,1,0,0,
        0,6,46,1,0,0,0,8,54,1,0,0,0,10,62,1,0,0,0,12,66,1,0,0,0,14,68,1,
        0,0,0,16,83,1,0,0,0,18,90,1,0,0,0,20,95,1,0,0,0,22,97,1,0,0,0,24,
        102,1,0,0,0,26,110,1,0,0,0,28,127,1,0,0,0,30,129,1,0,0,0,32,131,
        1,0,0,0,34,39,3,2,1,0,35,39,3,10,5,0,36,39,3,6,3,0,37,39,3,8,4,0,
        38,34,1,0,0,0,38,35,1,0,0,0,38,36,1,0,0,0,38,37,1,0,0,0,39,1,1,0,
        0,0,40,41,3,14,7,0,41,42,3,4,2,0,42,43,3,18,9,0,43,3,1,0,0,0,44,
        45,7,0,0,0,45,5,1,0,0,0,46,47,5,18,0,0,47,48,3,4,2,0,48,49,3,18,
        9,0,49,50,5,5,0,0,50,51,5,18,0,0,51,52,3,4,2,0,52,53,3,18,9,0,53,
        7,1,0,0,0,54,55,5,18,0,0,55,56,3,4,2,0,56,57,3,18,9,0,57,58,5,6,
        0,0,58,59,5,18,0,0,59,60,3,4,2,0,60,61,3,18,9,0,61,9,1,0,0,0,62,
        63,3,20,10,0,63,64,3,12,6,0,64,65,3,18,9,0,65,11,1,0,0,0,66,67,7,
        1,0,0,67,13,1,0,0,0,68,77,3,16,8,0,69,72,3,28,14,0,70,72,3,30,15,
        0,71,69,1,0,0,0,71,70,1,0,0,0,72,73,1,0,0,0,73,74,3,16,8,0,74,76,
        1,0,0,0,75,71,1,0,0,0,76,79,1,0,0,0,77,78,1,0,0,0,77,75,1,0,0,0,
        78,15,1,0,0,0,79,77,1,0,0,0,80,81,3,18,9,0,81,82,5,9,0,0,82,84,1,
        0,0,0,83,80,1,0,0,0,83,84,1,0,0,0,84,85,1,0,0,0,85,86,3,20,10,0,
        86,17,1,0,0,0,87,91,5,19,0,0,88,91,3,32,16,0,89,91,3,22,11,0,90,
        87,1,0,0,0,90,88,1,0,0,0,90,89,1,0,0,0,91,19,1,0,0,0,92,96,5,18,
        0,0,93,96,3,22,11,0,94,96,3,24,12,0,95,92,1,0,0,0,95,93,1,0,0,0,
        95,94,1,0,0,0,96,21,1,0,0,0,97,98,5,18,0,0,98,99,5,10,0,0,99,100,
        7,2,0,0,100,101,5,11,0,0,101,23,1,0,0,0,102,103,5,12,0,0,103,104,
        3,26,13,0,104,105,5,13,0,0,105,106,3,26,13,0,106,107,5,13,0,0,107,
        108,3,26,13,0,108,109,5,14,0,0,109,25,1,0,0,0,110,113,5,10,0,0,111,
        114,5,19,0,0,112,114,3,32,16,0,113,111,1,0,0,0,113,112,1,0,0,0,114,
        122,1,0,0,0,115,118,5,13,0,0,116,119,5,19,0,0,117,119,3,32,16,0,
        118,116,1,0,0,0,118,117,1,0,0,0,119,121,1,0,0,0,120,115,1,0,0,0,
        121,124,1,0,0,0,122,123,1,0,0,0,122,120,1,0,0,0,123,125,1,0,0,0,
        124,122,1,0,0,0,125,126,5,11,0,0,126,27,1,0,0,0,127,128,5,15,0,0,
        128,29,1,0,0,0,129,130,5,16,0,0,130,31,1,0,0,0,131,132,5,17,0,0,
        132,133,5,18,0,0,133,33,1,0,0,0,9,38,71,77,83,90,95,113,118,122
    ]

class TRUNCParser ( Parser ):

    grammarFileName = "TRUNC.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'<='", "'<'", "'>'", "'>='", "'and'", 
                     "'or'", "'=='", "'!='", "'*'", "'['", "']'", "'gm('", 
                     "','", "')'", "'+'", "'-'", "'_'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "IDV", "NUM", "COMM", "WS", 
                      "ALPHA", "DIGIT" ]

    RULE_trunc = 0
    RULE_ineq = 1
    RULE_inop = 2
    RULE_and_trunc = 3
    RULE_or_trunc = 4
    RULE_eq = 5
    RULE_eqop = 6
    RULE_lexpr = 7
    RULE_monom = 8
    RULE_const = 9
    RULE_var = 10
    RULE_idd = 11
    RULE_gm = 12
    RULE_list = 13
    RULE_sum = 14
    RULE_sub = 15
    RULE_par = 16

    ruleNames =  [ "trunc", "ineq", "inop", "and_trunc", "or_trunc", "eq", 
                   "eqop", "lexpr", "monom", "const", "var", "idd", "gm", 
                   "list", "sum", "sub", "par" ]

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
    IDV=18
    NUM=19
    COMM=20
    WS=21
    ALPHA=22
    DIGIT=23

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class TruncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ineq(self):
            return self.getTypedRuleContext(TRUNCParser.IneqContext,0)


        def eq(self):
            return self.getTypedRuleContext(TRUNCParser.EqContext,0)


        def and_trunc(self):
            return self.getTypedRuleContext(TRUNCParser.And_truncContext,0)


        def or_trunc(self):
            return self.getTypedRuleContext(TRUNCParser.Or_truncContext,0)


        def getRuleIndex(self):
            return TRUNCParser.RULE_trunc

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTrunc" ):
                listener.enterTrunc(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTrunc" ):
                listener.exitTrunc(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTrunc" ):
                return visitor.visitTrunc(self)
            else:
                return visitor.visitChildren(self)




    def trunc(self):

        localctx = TRUNCParser.TruncContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_trunc)
        try:
            self.state = 38
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 34
                self.ineq()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 35
                self.eq()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 36
                self.and_trunc()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 37
                self.or_trunc()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IneqContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def lexpr(self):
            return self.getTypedRuleContext(TRUNCParser.LexprContext,0)


        def inop(self):
            return self.getTypedRuleContext(TRUNCParser.InopContext,0)


        def const(self):
            return self.getTypedRuleContext(TRUNCParser.ConstContext,0)


        def getRuleIndex(self):
            return TRUNCParser.RULE_ineq

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIneq" ):
                listener.enterIneq(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIneq" ):
                listener.exitIneq(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIneq" ):
                return visitor.visitIneq(self)
            else:
                return visitor.visitChildren(self)




    def ineq(self):

        localctx = TRUNCParser.IneqContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_ineq)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 40
            self.lexpr()
            self.state = 41
            self.inop()
            self.state = 42
            self.const()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class InopContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return TRUNCParser.RULE_inop

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInop" ):
                listener.enterInop(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInop" ):
                listener.exitInop(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitInop" ):
                return visitor.visitInop(self)
            else:
                return visitor.visitChildren(self)




    def inop(self):

        localctx = TRUNCParser.InopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_inop)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 44
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << TRUNCParser.T__0) | (1 << TRUNCParser.T__1) | (1 << TRUNCParser.T__2) | (1 << TRUNCParser.T__3))) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class And_truncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self, i:int=None):
            if i is None:
                return self.getTokens(TRUNCParser.IDV)
            else:
                return self.getToken(TRUNCParser.IDV, i)

        def inop(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.InopContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.InopContext,i)


        def const(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.ConstContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.ConstContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_and_trunc

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAnd_trunc" ):
                listener.enterAnd_trunc(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAnd_trunc" ):
                listener.exitAnd_trunc(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAnd_trunc" ):
                return visitor.visitAnd_trunc(self)
            else:
                return visitor.visitChildren(self)




    def and_trunc(self):

        localctx = TRUNCParser.And_truncContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_and_trunc)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 46
            self.match(TRUNCParser.IDV)
            self.state = 47
            self.inop()
            self.state = 48
            self.const()
            self.state = 49
            self.match(TRUNCParser.T__4)
            self.state = 50
            self.match(TRUNCParser.IDV)
            self.state = 51
            self.inop()
            self.state = 52
            self.const()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Or_truncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self, i:int=None):
            if i is None:
                return self.getTokens(TRUNCParser.IDV)
            else:
                return self.getToken(TRUNCParser.IDV, i)

        def inop(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.InopContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.InopContext,i)


        def const(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.ConstContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.ConstContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_or_trunc

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOr_trunc" ):
                listener.enterOr_trunc(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOr_trunc" ):
                listener.exitOr_trunc(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOr_trunc" ):
                return visitor.visitOr_trunc(self)
            else:
                return visitor.visitChildren(self)




    def or_trunc(self):

        localctx = TRUNCParser.Or_truncContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_or_trunc)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 54
            self.match(TRUNCParser.IDV)
            self.state = 55
            self.inop()
            self.state = 56
            self.const()
            self.state = 57
            self.match(TRUNCParser.T__5)
            self.state = 58
            self.match(TRUNCParser.IDV)
            self.state = 59
            self.inop()
            self.state = 60
            self.const()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EqContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def var(self):
            return self.getTypedRuleContext(TRUNCParser.VarContext,0)


        def eqop(self):
            return self.getTypedRuleContext(TRUNCParser.EqopContext,0)


        def const(self):
            return self.getTypedRuleContext(TRUNCParser.ConstContext,0)


        def getRuleIndex(self):
            return TRUNCParser.RULE_eq

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEq" ):
                listener.enterEq(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEq" ):
                listener.exitEq(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitEq" ):
                return visitor.visitEq(self)
            else:
                return visitor.visitChildren(self)




    def eq(self):

        localctx = TRUNCParser.EqContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_eq)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 62
            self.var()
            self.state = 63
            self.eqop()
            self.state = 64
            self.const()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EqopContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return TRUNCParser.RULE_eqop

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEqop" ):
                listener.enterEqop(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEqop" ):
                listener.exitEqop(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitEqop" ):
                return visitor.visitEqop(self)
            else:
                return visitor.visitChildren(self)




    def eqop(self):

        localctx = TRUNCParser.EqopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_eqop)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 66
            _la = self._input.LA(1)
            if not(_la==TRUNCParser.T__6 or _la==TRUNCParser.T__7):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
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
                return self.getTypedRuleContexts(TRUNCParser.MonomContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.MonomContext,i)


        def sum_(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.SumContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.SumContext,i)


        def sub(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.SubContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.SubContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_lexpr

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

        localctx = TRUNCParser.LexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_lexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 68
            self.monom()
            self.state = 77
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 71
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [TRUNCParser.T__14]:
                        self.state = 69
                        self.sum_()
                        pass
                    elif token in [TRUNCParser.T__15]:
                        self.state = 70
                        self.sub()
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 73
                    self.monom() 
                self.state = 79
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

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

        def var(self):
            return self.getTypedRuleContext(TRUNCParser.VarContext,0)


        def const(self):
            return self.getTypedRuleContext(TRUNCParser.ConstContext,0)


        def getRuleIndex(self):
            return TRUNCParser.RULE_monom

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

        localctx = TRUNCParser.MonomContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_monom)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 83
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                self.state = 80
                self.const()
                self.state = 81
                self.match(TRUNCParser.T__8)


            self.state = 85
            self.var()
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

        def NUM(self):
            return self.getToken(TRUNCParser.NUM, 0)

        def par(self):
            return self.getTypedRuleContext(TRUNCParser.ParContext,0)


        def idd(self):
            return self.getTypedRuleContext(TRUNCParser.IddContext,0)


        def getRuleIndex(self):
            return TRUNCParser.RULE_const

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

        localctx = TRUNCParser.ConstContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_const)
        try:
            self.state = 90
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [TRUNCParser.NUM]:
                self.enterOuterAlt(localctx, 1)
                self.state = 87
                self.match(TRUNCParser.NUM)
                pass
            elif token in [TRUNCParser.T__16]:
                self.enterOuterAlt(localctx, 2)
                self.state = 88
                self.par()
                pass
            elif token in [TRUNCParser.IDV]:
                self.enterOuterAlt(localctx, 3)
                self.state = 89
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


    class VarContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self):
            return self.getToken(TRUNCParser.IDV, 0)

        def idd(self):
            return self.getTypedRuleContext(TRUNCParser.IddContext,0)


        def gm(self):
            return self.getTypedRuleContext(TRUNCParser.GmContext,0)


        def getRuleIndex(self):
            return TRUNCParser.RULE_var

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVar" ):
                listener.enterVar(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVar" ):
                listener.exitVar(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVar" ):
                return visitor.visitVar(self)
            else:
                return visitor.visitChildren(self)
            
        def _getText(self, data):
            if not self.idd() is None:
                return self.idd().getVar(data)
            else:
                return self.getText()
            

    def var(self):

        localctx = TRUNCParser.VarContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_var)
        try:
            self.state = 95
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 92
                self.match(TRUNCParser.IDV)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 93
                self.idd()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 94
                self.gm()
                pass


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
                return self.getTokens(TRUNCParser.IDV)
            else:
                return self.getToken(TRUNCParser.IDV, i)

        def NUM(self):
            return self.getToken(TRUNCParser.NUM, 0)

        def getRuleIndex(self):
            return TRUNCParser.RULE_idd

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

        def getVar(self, data):
            if self.IDV(1) is None:
                return self.getText()
            else:
                data_idx = int(data[self.IDV(1).getText()][0].item())
                return self.IDV(0).getText()+'['+str(data_idx)+']'  
    
        def getValue(self, data):
            data_name = self.IDV(0).getText()
            if not self.NUM() is None:
                data_idx = int(self.NUM().getText())
            elif not self.IDV(1) is None:
                data_idx = int(data[self.IDV(1).getText()][0].item())
            return data[data_name][data_idx]


    def idd(self):

        localctx = TRUNCParser.IddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_idd)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 97
            self.match(TRUNCParser.IDV)
            self.state = 98
            self.match(TRUNCParser.T__9)
            self.state = 99
            _la = self._input.LA(1)
            if not(_la==TRUNCParser.IDV or _la==TRUNCParser.NUM):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 100
            self.match(TRUNCParser.T__10)
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
                return self.getTypedRuleContexts(TRUNCParser.ListContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.ListContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_gm

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

        localctx = TRUNCParser.GmContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_gm)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 102
            self.match(TRUNCParser.T__11)
            self.state = 103
            self.list_()
            self.state = 104
            self.match(TRUNCParser.T__12)
            self.state = 105
            self.list_()
            self.state = 106
            self.match(TRUNCParser.T__12)
            self.state = 107
            self.list_()
            self.state = 108
            self.match(TRUNCParser.T__13)
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
                return self.getTokens(TRUNCParser.NUM)
            else:
                return self.getToken(TRUNCParser.NUM, i)

        def par(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.ParContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.ParContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_list

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

        def unpack(self, params_dict):
            str_list = self.getText()[1:-1].split(',')
            unpacked = torch.zeros(len(str_list))
            for i, elem in enumerate(str_list):
                if elem[0] == '_':
                    unpacked[i] = params_dict[elem[1:]]
                else:
                    unpacked[i] = float(elem)
            return unpacked



    def list_(self):

        localctx = TRUNCParser.ListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 110
            self.match(TRUNCParser.T__9)
            self.state = 113
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [TRUNCParser.NUM]:
                self.state = 111
                self.match(TRUNCParser.NUM)
                pass
            elif token in [TRUNCParser.T__16]:
                self.state = 112
                self.par()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 122
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,8,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 115
                    self.match(TRUNCParser.T__12)
                    self.state = 118
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [TRUNCParser.NUM]:
                        self.state = 116
                        self.match(TRUNCParser.NUM)
                        pass
                    elif token in [TRUNCParser.T__16]:
                        self.state = 117
                        self.par()
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 124
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,8,self._ctx)

            self.state = 125
            self.match(TRUNCParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SumContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return TRUNCParser.RULE_sum

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSum" ):
                listener.enterSum(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSum" ):
                listener.exitSum(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSum" ):
                return visitor.visitSum(self)
            else:
                return visitor.visitChildren(self)




    def sum_(self):

        localctx = TRUNCParser.SumContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_sum)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 127
            self.match(TRUNCParser.T__14)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SubContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return TRUNCParser.RULE_sub

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSub" ):
                listener.enterSub(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSub" ):
                listener.exitSub(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSub" ):
                return visitor.visitSub(self)
            else:
                return visitor.visitChildren(self)




    def sub(self):

        localctx = TRUNCParser.SubContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_sub)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 129
            self.match(TRUNCParser.T__15)
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
            return self.getToken(TRUNCParser.IDV, 0)

        def getRuleIndex(self):
            return TRUNCParser.RULE_par

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

        def getValue(self, params_dict):
            name = self.getText()
            name = name[1:]
            return params_dict[name]


    def par(self):

        localctx = TRUNCParser.ParContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_par)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 131
            self.match(TRUNCParser.T__16)
            self.state = 132
            self.match(TRUNCParser.IDV)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





