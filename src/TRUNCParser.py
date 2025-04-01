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
        4,1,23,156,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,1,0,1,0,1,0,1,
        0,3,0,43,8,0,1,1,1,1,1,1,1,1,1,2,1,2,1,3,1,3,1,3,1,3,1,3,1,3,1,3,
        1,3,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,5,1,5,1,5,1,5,1,6,1,6,1,7,
        1,7,1,7,1,7,5,7,77,8,7,10,7,12,7,80,9,7,1,8,1,8,1,8,3,8,85,8,8,1,
        8,1,8,1,9,1,9,1,9,1,9,5,9,93,8,9,10,9,12,9,96,9,9,1,10,1,10,1,10,
        3,10,101,8,10,1,11,1,11,1,11,3,11,106,8,11,1,12,1,12,1,12,3,12,111,
        8,12,1,13,1,13,1,13,3,13,116,8,13,1,14,1,14,1,14,1,14,3,14,122,8,
        14,1,14,1,14,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,16,1,16,1,
        16,3,16,137,8,16,1,16,1,16,1,16,3,16,142,8,16,5,16,144,8,16,10,16,
        12,16,147,9,16,1,16,1,16,1,17,1,17,1,17,1,18,1,18,1,18,3,78,94,145,
        0,19,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,0,3,1,0,
        1,4,1,0,7,8,1,0,15,16,153,0,42,1,0,0,0,2,44,1,0,0,0,4,48,1,0,0,0,
        6,50,1,0,0,0,8,58,1,0,0,0,10,66,1,0,0,0,12,70,1,0,0,0,14,72,1,0,
        0,0,16,84,1,0,0,0,18,88,1,0,0,0,20,100,1,0,0,0,22,105,1,0,0,0,24,
        110,1,0,0,0,26,115,1,0,0,0,28,117,1,0,0,0,30,125,1,0,0,0,32,133,
        1,0,0,0,34,150,1,0,0,0,36,153,1,0,0,0,38,43,3,2,1,0,39,43,3,10,5,
        0,40,43,3,6,3,0,41,43,3,8,4,0,42,38,1,0,0,0,42,39,1,0,0,0,42,40,
        1,0,0,0,42,41,1,0,0,0,43,1,1,0,0,0,44,45,3,14,7,0,45,46,3,4,2,0,
        46,47,3,18,9,0,47,3,1,0,0,0,48,49,7,0,0,0,49,5,1,0,0,0,50,51,5,18,
        0,0,51,52,3,4,2,0,52,53,3,18,9,0,53,54,5,5,0,0,54,55,5,18,0,0,55,
        56,3,4,2,0,56,57,3,18,9,0,57,7,1,0,0,0,58,59,5,18,0,0,59,60,3,4,
        2,0,60,61,3,18,9,0,61,62,5,6,0,0,62,63,5,18,0,0,63,64,3,4,2,0,64,
        65,3,18,9,0,65,9,1,0,0,0,66,67,3,26,13,0,67,68,3,12,6,0,68,69,3,
        18,9,0,69,11,1,0,0,0,70,71,7,1,0,0,71,13,1,0,0,0,72,78,3,16,8,0,
        73,74,3,36,18,0,74,75,3,16,8,0,75,77,1,0,0,0,76,73,1,0,0,0,77,80,
        1,0,0,0,78,79,1,0,0,0,78,76,1,0,0,0,79,15,1,0,0,0,80,78,1,0,0,0,
        81,82,3,22,11,0,82,83,5,17,0,0,83,85,1,0,0,0,84,81,1,0,0,0,84,85,
        1,0,0,0,85,86,1,0,0,0,86,87,3,26,13,0,87,17,1,0,0,0,88,94,3,22,11,
        0,89,90,3,36,18,0,90,91,3,20,10,0,91,93,1,0,0,0,92,89,1,0,0,0,93,
        96,1,0,0,0,94,95,1,0,0,0,94,92,1,0,0,0,95,19,1,0,0,0,96,94,1,0,0,
        0,97,101,5,19,0,0,98,101,3,34,17,0,99,101,3,28,14,0,100,97,1,0,0,
        0,100,98,1,0,0,0,100,99,1,0,0,0,101,21,1,0,0,0,102,106,3,24,12,0,
        103,106,3,34,17,0,104,106,3,28,14,0,105,102,1,0,0,0,105,103,1,0,
        0,0,105,104,1,0,0,0,106,23,1,0,0,0,107,111,5,19,0,0,108,109,5,16,
        0,0,109,111,5,19,0,0,110,107,1,0,0,0,110,108,1,0,0,0,111,25,1,0,
        0,0,112,116,5,18,0,0,113,116,3,28,14,0,114,116,3,30,15,0,115,112,
        1,0,0,0,115,113,1,0,0,0,115,114,1,0,0,0,116,27,1,0,0,0,117,118,5,
        18,0,0,118,121,5,9,0,0,119,122,3,24,12,0,120,122,5,18,0,0,121,119,
        1,0,0,0,121,120,1,0,0,0,122,123,1,0,0,0,123,124,5,10,0,0,124,29,
        1,0,0,0,125,126,5,11,0,0,126,127,3,32,16,0,127,128,5,12,0,0,128,
        129,3,32,16,0,129,130,5,12,0,0,130,131,3,32,16,0,131,132,5,13,0,
        0,132,31,1,0,0,0,133,136,5,9,0,0,134,137,3,24,12,0,135,137,3,34,
        17,0,136,134,1,0,0,0,136,135,1,0,0,0,137,145,1,0,0,0,138,141,5,12,
        0,0,139,142,3,24,12,0,140,142,3,34,17,0,141,139,1,0,0,0,141,140,
        1,0,0,0,142,144,1,0,0,0,143,138,1,0,0,0,144,147,1,0,0,0,145,146,
        1,0,0,0,145,143,1,0,0,0,146,148,1,0,0,0,147,145,1,0,0,0,148,149,
        5,10,0,0,149,33,1,0,0,0,150,151,5,14,0,0,151,152,5,18,0,0,152,35,
        1,0,0,0,153,154,7,2,0,0,154,37,1,0,0,0,12,42,78,84,94,100,105,110,
        115,121,136,141,145
    ]

class TRUNCParser ( Parser ):

    grammarFileName = "TRUNC.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'<='", "'<'", "'>'", "'>='", "'and'", 
                     "'or'", "'=='", "'!='", "'['", "']'", "'gm('", "','", 
                     "')'", "'_'", "'+'", "'-'", "'*'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "SUM", "SUB", 
                      "MUL", "IDV", "POSNUM", "COMM", "WS", "ALPHA", "DIGIT" ]

    RULE_trunc = 0
    RULE_ineq = 1
    RULE_inop = 2
    RULE_and_trunc = 3
    RULE_or_trunc = 4
    RULE_eq = 5
    RULE_eqop = 6
    RULE_lexpr = 7
    RULE_monom = 8
    RULE_const_expr = 9
    RULE_pos_const = 10
    RULE_const = 11
    RULE_num = 12
    RULE_var = 13
    RULE_idd = 14
    RULE_gm = 15
    RULE_list = 16
    RULE_par = 17
    RULE_aop = 18

    ruleNames =  [ "trunc", "ineq", "inop", "and_trunc", "or_trunc", "eq", 
                   "eqop", "lexpr", "monom", "const_expr", "pos_const", 
                   "const", "num", "var", "idd", "gm", "list", "par", "aop" ]

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
    SUM=15
    SUB=16
    MUL=17
    IDV=18
    POSNUM=19
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
            self.state = 42
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 38
                self.ineq()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 39
                self.eq()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 40
                self.and_trunc()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 41
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


        def const_expr(self):
            return self.getTypedRuleContext(TRUNCParser.Const_exprContext,0)


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
            self.state = 44
            self.lexpr()
            self.state = 45
            self.inop()
            self.state = 46
            self.const_expr()
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
            self.state = 48
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


        def const_expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.Const_exprContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.Const_exprContext,i)


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
            self.state = 50
            self.match(TRUNCParser.IDV)
            self.state = 51
            self.inop()
            self.state = 52
            self.const_expr()
            self.state = 53
            self.match(TRUNCParser.T__4)
            self.state = 54
            self.match(TRUNCParser.IDV)
            self.state = 55
            self.inop()
            self.state = 56
            self.const_expr()
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


        def const_expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.Const_exprContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.Const_exprContext,i)


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
            self.state = 58
            self.match(TRUNCParser.IDV)
            self.state = 59
            self.inop()
            self.state = 60
            self.const_expr()
            self.state = 61
            self.match(TRUNCParser.T__5)
            self.state = 62
            self.match(TRUNCParser.IDV)
            self.state = 63
            self.inop()
            self.state = 64
            self.const_expr()
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


        def const_expr(self):
            return self.getTypedRuleContext(TRUNCParser.Const_exprContext,0)


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
            self.state = 66
            self.var()
            self.state = 67
            self.eqop()
            self.state = 68
            self.const_expr()
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
            self.state = 70
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


        def aop(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.AopContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.AopContext,i)


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
            self.state = 72
            self.monom()
            self.state = 78
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,1,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 73
                    self.aop()
                    self.state = 74
                    self.monom() 
                self.state = 80
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,1,self._ctx)

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


        def MUL(self):
            return self.getToken(TRUNCParser.MUL, 0)

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
            self.state = 84
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                self.state = 81
                self.const()
                self.state = 82
                self.match(TRUNCParser.MUL)


            self.state = 86
            self.var()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Const_exprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def const(self):
            return self.getTypedRuleContext(TRUNCParser.ConstContext,0)


        def aop(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.AopContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.AopContext,i)


        def pos_const(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.Pos_constContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.Pos_constContext,i)


        def getRuleIndex(self):
            return TRUNCParser.RULE_const_expr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterConst_expr" ):
                listener.enterConst_expr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitConst_expr" ):
                listener.exitConst_expr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitConst_expr" ):
                return visitor.visitConst_expr(self)
            else:
                return visitor.visitChildren(self)




    def const_expr(self):

        localctx = TRUNCParser.Const_exprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_const_expr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 88
            self.const()
            self.state = 94
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,3,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 89
                    self.aop()
                    self.state = 90
                    self.pos_const() 
                self.state = 96
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,3,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Pos_constContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def POSNUM(self):
            return self.getToken(TRUNCParser.POSNUM, 0)

        def par(self):
            return self.getTypedRuleContext(TRUNCParser.ParContext,0)


        def idd(self):
            return self.getTypedRuleContext(TRUNCParser.IddContext,0)


        def getRuleIndex(self):
            return TRUNCParser.RULE_pos_const

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPos_const" ):
                listener.enterPos_const(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPos_const" ):
                listener.exitPos_const(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPos_const" ):
                return visitor.visitPos_const(self)
            else:
                return visitor.visitChildren(self)




    def pos_const(self):

        localctx = TRUNCParser.Pos_constContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_pos_const)
        try:
            self.state = 100
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [TRUNCParser.POSNUM]:
                self.enterOuterAlt(localctx, 1)
                self.state = 97
                self.match(TRUNCParser.POSNUM)
                pass
            elif token in [TRUNCParser.T__13]:
                self.enterOuterAlt(localctx, 2)
                self.state = 98
                self.par()
                pass
            elif token in [TRUNCParser.IDV]:
                self.enterOuterAlt(localctx, 3)
                self.state = 99
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


    class ConstContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def num(self):
            return self.getTypedRuleContext(TRUNCParser.NumContext,0)


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
        self.enterRule(localctx, 22, self.RULE_const)
        try:
            self.state = 105
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [TRUNCParser.SUB, TRUNCParser.POSNUM]:
                self.enterOuterAlt(localctx, 1)
                self.state = 102
                self.num()
                pass
            elif token in [TRUNCParser.T__13]:
                self.enterOuterAlt(localctx, 2)
                self.state = 103
                self.par()
                pass
            elif token in [TRUNCParser.IDV]:
                self.enterOuterAlt(localctx, 3)
                self.state = 104
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


    class NumContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def POSNUM(self):
            return self.getToken(TRUNCParser.POSNUM, 0)

        def SUB(self):
            return self.getToken(TRUNCParser.SUB, 0)

        def getRuleIndex(self):
            return TRUNCParser.RULE_num

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterNum" ):
                listener.enterNum(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitNum" ):
                listener.exitNum(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitNum" ):
                return visitor.visitNum(self)
            else:
                return visitor.visitChildren(self)




    def num(self):

        localctx = TRUNCParser.NumContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_num)
        try:
            self.state = 110
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [TRUNCParser.POSNUM]:
                self.enterOuterAlt(localctx, 1)
                self.state = 107
                self.match(TRUNCParser.POSNUM)
                pass
            elif token in [TRUNCParser.SUB]:
                self.enterOuterAlt(localctx, 2)
                self.state = 108
                self.match(TRUNCParser.SUB)
                self.state = 109
                self.match(TRUNCParser.POSNUM)
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
            if self.idd():
                return self.idd().getVar(data)
            else:
                return self.getText()




    def var(self):

        localctx = TRUNCParser.VarContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_var)
        try:
            self.state = 115
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 112
                self.match(TRUNCParser.IDV)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 113
                self.idd()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 114
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

        def num(self):
            return self.getTypedRuleContext(TRUNCParser.NumContext,0)


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
            if not self.num() is None:
                data_idx = int(self.num().getText())
            elif not self.IDV(1) is None:
                data_idx = int(data[self.IDV(1).getText()][0].item())
            return data[data_name][data_idx]



    def idd(self):

        localctx = TRUNCParser.IddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_idd)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 117
            self.match(TRUNCParser.IDV)
            self.state = 118
            self.match(TRUNCParser.T__8)
            self.state = 121
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [TRUNCParser.SUB, TRUNCParser.POSNUM]:
                self.state = 119
                self.num()
                pass
            elif token in [TRUNCParser.IDV]:
                self.state = 120
                self.match(TRUNCParser.IDV)
                pass
            else:
                raise NoViableAltException(self)

            self.state = 123
            self.match(TRUNCParser.T__9)
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
        self.enterRule(localctx, 30, self.RULE_gm)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 125
            self.match(TRUNCParser.T__10)
            self.state = 126
            self.list_()
            self.state = 127
            self.match(TRUNCParser.T__11)
            self.state = 128
            self.list_()
            self.state = 129
            self.match(TRUNCParser.T__11)
            self.state = 130
            self.list_()
            self.state = 131
            self.match(TRUNCParser.T__12)
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

        def num(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TRUNCParser.NumContext)
            else:
                return self.getTypedRuleContext(TRUNCParser.NumContext,i)


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
        self.enterRule(localctx, 32, self.RULE_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 133
            self.match(TRUNCParser.T__8)
            self.state = 136
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [TRUNCParser.SUB, TRUNCParser.POSNUM]:
                self.state = 134
                self.num()
                pass
            elif token in [TRUNCParser.T__13]:
                self.state = 135
                self.par()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 145
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,11,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 138
                    self.match(TRUNCParser.T__11)
                    self.state = 141
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [TRUNCParser.SUB, TRUNCParser.POSNUM]:
                        self.state = 139
                        self.num()
                        pass
                    elif token in [TRUNCParser.T__13]:
                        self.state = 140
                        self.par()
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 147
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,11,self._ctx)

            self.state = 148
            self.match(TRUNCParser.T__9)
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
        self.enterRule(localctx, 34, self.RULE_par)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 150
            self.match(TRUNCParser.T__13)
            self.state = 151
            self.match(TRUNCParser.IDV)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AopContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def SUM(self):
            return self.getToken(TRUNCParser.SUM, 0)

        def SUB(self):
            return self.getToken(TRUNCParser.SUB, 0)

        def getRuleIndex(self):
            return TRUNCParser.RULE_aop

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAop" ):
                listener.enterAop(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAop" ):
                listener.exitAop(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAop" ):
                return visitor.visitAop(self)
            else:
                return visitor.visitChildren(self)




    def aop(self):

        localctx = TRUNCParser.AopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_aop)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 153
            _la = self._input.LA(1)
            if not(_la==TRUNCParser.SUM or _la==TRUNCParser.SUB):
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





