# Generated from grammars/ASGMT.g4 by ANTLR 4.10
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,19,164,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,3,0,41,8,0,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,1,52,8,1,10,1,12,1,55,9,1,1,
        2,1,2,1,2,1,2,1,2,3,2,62,8,2,1,3,1,3,1,3,1,3,1,3,1,4,1,4,1,4,1,4,
        1,4,1,4,1,4,1,4,1,4,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,6,1,6,1,6,1,6,
        5,6,89,8,6,10,6,12,6,92,9,6,1,6,1,6,1,7,1,7,1,7,1,7,1,7,1,7,1,7,
        1,8,1,8,3,8,105,8,8,1,8,5,8,108,8,8,10,8,12,8,111,9,8,1,9,1,9,1,
        9,3,9,116,8,9,1,9,1,9,1,10,3,10,121,8,10,1,10,1,10,3,10,125,8,10,
        1,10,1,10,3,10,129,8,10,1,10,3,10,132,8,10,1,11,1,11,3,11,136,8,
        11,1,12,1,12,1,12,1,12,1,12,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,
        13,1,14,1,14,1,14,1,14,5,14,155,8,14,10,14,12,14,158,9,14,1,14,1,
        14,1,15,1,15,1,15,2,109,156,1,2,16,0,2,4,6,8,10,12,14,16,18,20,22,
        24,26,28,30,0,1,1,0,15,16,165,0,40,1,0,0,0,2,42,1,0,0,0,4,61,1,0,
        0,0,6,63,1,0,0,0,8,68,1,0,0,0,10,77,1,0,0,0,12,84,1,0,0,0,14,95,
        1,0,0,0,16,102,1,0,0,0,18,115,1,0,0,0,20,131,1,0,0,0,22,135,1,0,
        0,0,24,137,1,0,0,0,26,142,1,0,0,0,28,150,1,0,0,0,30,161,1,0,0,0,
        32,33,3,22,11,0,33,34,5,1,0,0,34,35,3,16,8,0,35,41,1,0,0,0,36,37,
        3,22,11,0,37,38,5,1,0,0,38,39,3,2,1,0,39,41,1,0,0,0,40,32,1,0,0,
        0,40,36,1,0,0,0,41,1,1,0,0,0,42,43,6,1,-1,0,43,44,3,4,2,0,44,53,
        1,0,0,0,45,46,10,3,0,0,46,47,5,14,0,0,47,52,3,4,2,0,48,49,10,2,0,
        0,49,50,5,2,0,0,50,52,3,4,2,0,51,45,1,0,0,0,51,48,1,0,0,0,52,55,
        1,0,0,0,53,51,1,0,0,0,53,54,1,0,0,0,54,3,1,0,0,0,55,53,1,0,0,0,56,
        62,3,6,3,0,57,62,3,14,7,0,58,62,5,15,0,0,59,62,3,8,4,0,60,62,3,10,
        5,0,61,56,1,0,0,0,61,57,1,0,0,0,61,58,1,0,0,0,61,59,1,0,0,0,61,60,
        1,0,0,0,62,5,1,0,0,0,63,64,5,11,0,0,64,65,5,3,0,0,65,66,5,15,0,0,
        66,67,5,4,0,0,67,7,1,0,0,0,68,69,5,13,0,0,69,70,5,3,0,0,70,71,3,
        12,6,0,71,72,5,5,0,0,72,73,3,12,6,0,73,74,5,5,0,0,74,75,3,12,6,0,
        75,76,5,4,0,0,76,9,1,0,0,0,77,78,5,12,0,0,78,79,5,3,0,0,79,80,3,
        12,6,0,80,81,5,5,0,0,81,82,3,12,6,0,82,83,5,4,0,0,83,11,1,0,0,0,
        84,85,5,6,0,0,85,90,3,28,14,0,86,87,5,5,0,0,87,89,3,28,14,0,88,86,
        1,0,0,0,89,92,1,0,0,0,90,88,1,0,0,0,90,91,1,0,0,0,91,93,1,0,0,0,
        92,90,1,0,0,0,93,94,5,7,0,0,94,13,1,0,0,0,95,96,5,15,0,0,96,97,5,
        6,0,0,97,98,7,0,0,0,98,99,5,5,0,0,99,100,7,0,0,0,100,101,5,7,0,0,
        101,15,1,0,0,0,102,109,3,18,9,0,103,105,5,2,0,0,104,103,1,0,0,0,
        104,105,1,0,0,0,105,106,1,0,0,0,106,108,3,18,9,0,107,104,1,0,0,0,
        108,111,1,0,0,0,109,110,1,0,0,0,109,107,1,0,0,0,110,17,1,0,0,0,111,
        109,1,0,0,0,112,113,3,20,10,0,113,114,5,8,0,0,114,116,1,0,0,0,115,
        112,1,0,0,0,115,116,1,0,0,0,116,117,1,0,0,0,117,118,3,20,10,0,118,
        19,1,0,0,0,119,121,3,30,15,0,120,119,1,0,0,0,120,121,1,0,0,0,121,
        122,1,0,0,0,122,132,5,16,0,0,123,125,3,30,15,0,124,123,1,0,0,0,124,
        125,1,0,0,0,125,126,1,0,0,0,126,132,3,22,11,0,127,129,3,30,15,0,
        128,127,1,0,0,0,128,129,1,0,0,0,129,130,1,0,0,0,130,132,3,26,13,
        0,131,120,1,0,0,0,131,124,1,0,0,0,131,128,1,0,0,0,132,21,1,0,0,0,
        133,136,5,15,0,0,134,136,3,24,12,0,135,133,1,0,0,0,135,134,1,0,0,
        0,136,23,1,0,0,0,137,138,5,15,0,0,138,139,5,6,0,0,139,140,7,0,0,
        0,140,141,5,7,0,0,141,25,1,0,0,0,142,143,5,9,0,0,143,144,3,28,14,
        0,144,145,5,5,0,0,145,146,3,28,14,0,146,147,5,5,0,0,147,148,3,28,
        14,0,148,149,5,4,0,0,149,27,1,0,0,0,150,151,5,6,0,0,151,156,5,16,
        0,0,152,153,5,5,0,0,153,155,5,16,0,0,154,152,1,0,0,0,155,158,1,0,
        0,0,156,157,1,0,0,0,156,154,1,0,0,0,157,159,1,0,0,0,158,156,1,0,
        0,0,159,160,5,7,0,0,160,29,1,0,0,0,161,162,5,10,0,0,162,31,1,0,0,
        0,14,40,51,53,61,90,104,109,115,120,124,128,131,135,156
    ]

class ASGMTParser ( Parser ):

    grammarFileName = "ASGMT.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'='", "'+'", "'('", "')'", "','", "'['", 
                     "']'", "'*'", "'gm('", "'-'", "'transp'", "'matrix_gm_full'", 
                     "'matrix_gm'", "'@'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "TRANSP", "MATRIX_GM_FULL", 
                      "MATRIX_GM", "MATMUL", "IDV", "NUM", "COMM", "WS", 
                      "DIGIT" ]

    RULE_assignment = 0
    RULE_mat_expr = 1
    RULE_mat_atom = 2
    RULE_transp = 3
    RULE_matrix_gm = 4
    RULE_matrix_gm_full = 5
    RULE_mlist = 6
    RULE_mat_idd = 7
    RULE_add = 8
    RULE_add_term = 9
    RULE_term = 10
    RULE_symvars = 11
    RULE_idd = 12
    RULE_gm = 13
    RULE_list = 14
    RULE_sub = 15

    ruleNames =  [ "assignment", "mat_expr", "mat_atom", "transp", "matrix_gm", 
                   "matrix_gm_full", "mlist", "mat_idd", "add", "add_term", 
                   "term", "symvars", "idd", "gm", "list", "sub" ]

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
    TRANSP=11
    MATRIX_GM_FULL=12
    MATRIX_GM=13
    MATMUL=14
    IDV=15
    NUM=16
    COMM=17
    WS=18
    DIGIT=19

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class AssignmentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def symvars(self):
            return self.getTypedRuleContext(ASGMTParser.SymvarsContext,0)


        def add(self):
            return self.getTypedRuleContext(ASGMTParser.AddContext,0)


        def mat_expr(self):
            return self.getTypedRuleContext(ASGMTParser.Mat_exprContext,0)


        def getRuleIndex(self):
            return ASGMTParser.RULE_assignment

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

        localctx = ASGMTParser.AssignmentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_assignment)
        try:
            self.state = 40
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 32
                self.symvars()
                self.state = 33
                self.match(ASGMTParser.T__0)
                self.state = 34
                self.add()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 36
                self.symvars()
                self.state = 37
                self.match(ASGMTParser.T__0)
                self.state = 38
                self.mat_expr(0)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Mat_exprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def mat_atom(self):
            return self.getTypedRuleContext(ASGMTParser.Mat_atomContext,0)


        def mat_expr(self):
            return self.getTypedRuleContext(ASGMTParser.Mat_exprContext,0)


        def MATMUL(self):
            return self.getToken(ASGMTParser.MATMUL, 0)

        def getRuleIndex(self):
            return ASGMTParser.RULE_mat_expr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMat_expr" ):
                listener.enterMat_expr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMat_expr" ):
                listener.exitMat_expr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMat_expr" ):
                return visitor.visitMat_expr(self)
            else:
                return visitor.visitChildren(self)



    def mat_expr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = ASGMTParser.Mat_exprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 2
        self.enterRecursionRule(localctx, 2, self.RULE_mat_expr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 43
            self.mat_atom()
            self._ctx.stop = self._input.LT(-1)
            self.state = 53
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 51
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
                    if la_ == 1:
                        localctx = ASGMTParser.Mat_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_mat_expr)
                        self.state = 45
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 3)")
                        self.state = 46
                        self.match(ASGMTParser.MATMUL)
                        self.state = 47
                        self.mat_atom()
                        pass

                    elif la_ == 2:
                        localctx = ASGMTParser.Mat_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_mat_expr)
                        self.state = 48
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                        self.state = 49
                        self.match(ASGMTParser.T__1)
                        self.state = 50
                        self.mat_atom()
                        pass

             
                self.state = 55
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class Mat_atomContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def transp(self):
            return self.getTypedRuleContext(ASGMTParser.TranspContext,0)


        def mat_idd(self):
            return self.getTypedRuleContext(ASGMTParser.Mat_iddContext,0)


        def IDV(self):
            return self.getToken(ASGMTParser.IDV, 0)

        def matrix_gm(self):
            return self.getTypedRuleContext(ASGMTParser.Matrix_gmContext,0)


        def matrix_gm_full(self):
            return self.getTypedRuleContext(ASGMTParser.Matrix_gm_fullContext,0)


        def getRuleIndex(self):
            return ASGMTParser.RULE_mat_atom

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMat_atom" ):
                listener.enterMat_atom(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMat_atom" ):
                listener.exitMat_atom(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMat_atom" ):
                return visitor.visitMat_atom(self)
            else:
                return visitor.visitChildren(self)




    def mat_atom(self):

        localctx = ASGMTParser.Mat_atomContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_mat_atom)
        try:
            self.state = 61
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 56
                self.transp()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 57
                self.mat_idd()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 58
                self.match(ASGMTParser.IDV)
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 59
                self.matrix_gm()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 60
                self.matrix_gm_full()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TranspContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TRANSP(self):
            return self.getToken(ASGMTParser.TRANSP, 0)

        def IDV(self):
            return self.getToken(ASGMTParser.IDV, 0)

        def getRuleIndex(self):
            return ASGMTParser.RULE_transp

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTransp" ):
                listener.enterTransp(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTransp" ):
                listener.exitTransp(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTransp" ):
                return visitor.visitTransp(self)
            else:
                return visitor.visitChildren(self)




    def transp(self):

        localctx = ASGMTParser.TranspContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_transp)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 63
            self.match(ASGMTParser.TRANSP)
            self.state = 64
            self.match(ASGMTParser.T__2)
            self.state = 65
            self.match(ASGMTParser.IDV)
            self.state = 66
            self.match(ASGMTParser.T__3)
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
            return self.getToken(ASGMTParser.MATRIX_GM, 0)

        def mlist(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.MlistContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.MlistContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_matrix_gm

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

        localctx = ASGMTParser.Matrix_gmContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_matrix_gm)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 68
            self.match(ASGMTParser.MATRIX_GM)
            self.state = 69
            self.match(ASGMTParser.T__2)
            self.state = 70
            self.mlist()
            self.state = 71
            self.match(ASGMTParser.T__4)
            self.state = 72
            self.mlist()
            self.state = 73
            self.match(ASGMTParser.T__4)
            self.state = 74
            self.mlist()
            self.state = 75
            self.match(ASGMTParser.T__3)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Matrix_gm_fullContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def MATRIX_GM_FULL(self):
            return self.getToken(ASGMTParser.MATRIX_GM_FULL, 0)

        def mlist(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.MlistContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.MlistContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_matrix_gm_full

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatrix_gm_full" ):
                listener.enterMatrix_gm_full(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatrix_gm_full" ):
                listener.exitMatrix_gm_full(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMatrix_gm_full" ):
                return visitor.visitMatrix_gm_full(self)
            else:
                return visitor.visitChildren(self)




    def matrix_gm_full(self):

        localctx = ASGMTParser.Matrix_gm_fullContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_matrix_gm_full)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 77
            self.match(ASGMTParser.MATRIX_GM_FULL)
            self.state = 78
            self.match(ASGMTParser.T__2)
            self.state = 79
            self.mlist()
            self.state = 80
            self.match(ASGMTParser.T__4)
            self.state = 81
            self.mlist()
            self.state = 82
            self.match(ASGMTParser.T__3)
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
                return self.getTypedRuleContexts(ASGMTParser.ListContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.ListContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_mlist

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

        localctx = ASGMTParser.MlistContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_mlist)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 84
            self.match(ASGMTParser.T__5)
            self.state = 85
            self.list_()
            self.state = 90
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==ASGMTParser.T__4:
                self.state = 86
                self.match(ASGMTParser.T__4)
                self.state = 87
                self.list_()
                self.state = 92
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 93
            self.match(ASGMTParser.T__6)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Mat_iddContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDV(self, i:int=None):
            if i is None:
                return self.getTokens(ASGMTParser.IDV)
            else:
                return self.getToken(ASGMTParser.IDV, i)

        def NUM(self, i:int=None):
            if i is None:
                return self.getTokens(ASGMTParser.NUM)
            else:
                return self.getToken(ASGMTParser.NUM, i)

        def getRuleIndex(self):
            return ASGMTParser.RULE_mat_idd

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMat_idd" ):
                listener.enterMat_idd(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMat_idd" ):
                listener.exitMat_idd(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMat_idd" ):
                return visitor.visitMat_idd(self)
            else:
                return visitor.visitChildren(self)




    def mat_idd(self):

        localctx = ASGMTParser.Mat_iddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_mat_idd)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 95
            self.match(ASGMTParser.IDV)
            self.state = 96
            self.match(ASGMTParser.T__5)
            self.state = 97
            _la = self._input.LA(1)
            if not(_la==ASGMTParser.IDV or _la==ASGMTParser.NUM):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 98
            self.match(ASGMTParser.T__4)
            self.state = 99
            _la = self._input.LA(1)
            if not(_la==ASGMTParser.IDV or _la==ASGMTParser.NUM):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 100
            self.match(ASGMTParser.T__6)
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
                return self.getTypedRuleContexts(ASGMTParser.Add_termContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.Add_termContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_add

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

        localctx = ASGMTParser.AddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_add)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 102
            self.add_term()
            self.state = 109
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,6,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 104
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==ASGMTParser.T__1:
                        self.state = 103
                        self.match(ASGMTParser.T__1)


                    self.state = 106
                    self.add_term() 
                self.state = 111
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,6,self._ctx)

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

        def term(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.TermContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.TermContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_add_term

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

        localctx = ASGMTParser.Add_termContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_add_term)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 115
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
            if la_ == 1:
                self.state = 112
                self.term()
                self.state = 113
                self.match(ASGMTParser.T__7)


            self.state = 117
            self.term()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TermContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUM(self):
            return self.getToken(ASGMTParser.NUM, 0)

        def sub(self):
            return self.getTypedRuleContext(ASGMTParser.SubContext,0)


        def symvars(self):
            return self.getTypedRuleContext(ASGMTParser.SymvarsContext,0)


        def gm(self):
            return self.getTypedRuleContext(ASGMTParser.GmContext,0)


        def getRuleIndex(self):
            return ASGMTParser.RULE_term

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTerm" ):
                listener.enterTerm(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTerm" ):
                listener.exitTerm(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTerm" ):
                return visitor.visitTerm(self)
            else:
                return visitor.visitChildren(self)




    def term(self):

        localctx = ASGMTParser.TermContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_term)
        self._la = 0 # Token type
        try:
            self.state = 131
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,11,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 120
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__9:
                    self.state = 119
                    self.sub()


                self.state = 122
                self.match(ASGMTParser.NUM)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 124
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__9:
                    self.state = 123
                    self.sub()


                self.state = 126
                self.symvars()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 128
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__9:
                    self.state = 127
                    self.sub()


                self.state = 130
                self.gm()
                pass


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
            return self.getToken(ASGMTParser.IDV, 0)

        def idd(self):
            return self.getTypedRuleContext(ASGMTParser.IddContext,0)


        def getRuleIndex(self):
            return ASGMTParser.RULE_symvars

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

        localctx = ASGMTParser.SymvarsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_symvars)
        try:
            self.state = 135
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,12,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 133
                self.match(ASGMTParser.IDV)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 134
                self.idd()
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
                return self.getTokens(ASGMTParser.IDV)
            else:
                return self.getToken(ASGMTParser.IDV, i)

        def NUM(self):
            return self.getToken(ASGMTParser.NUM, 0)

        def getRuleIndex(self):
            return ASGMTParser.RULE_idd

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

        localctx = ASGMTParser.IddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_idd)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 137
            self.match(ASGMTParser.IDV)
            self.state = 138
            self.match(ASGMTParser.T__5)
            self.state = 139
            _la = self._input.LA(1)
            if not(_la==ASGMTParser.IDV or _la==ASGMTParser.NUM):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 140
            self.match(ASGMTParser.T__6)
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
                return self.getTypedRuleContexts(ASGMTParser.ListContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.ListContext,i)


        def getRuleIndex(self):
            return ASGMTParser.RULE_gm

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

        localctx = ASGMTParser.GmContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_gm)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 142
            self.match(ASGMTParser.T__8)
            self.state = 143
            self.list_()
            self.state = 144
            self.match(ASGMTParser.T__4)
            self.state = 145
            self.list_()
            self.state = 146
            self.match(ASGMTParser.T__4)
            self.state = 147
            self.list_()
            self.state = 148
            self.match(ASGMTParser.T__3)
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
                return self.getTokens(ASGMTParser.NUM)
            else:
                return self.getToken(ASGMTParser.NUM, i)

        def getRuleIndex(self):
            return ASGMTParser.RULE_list

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

        localctx = ASGMTParser.ListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 150
            self.match(ASGMTParser.T__5)
            self.state = 151
            self.match(ASGMTParser.NUM)
            self.state = 156
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,13,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 152
                    self.match(ASGMTParser.T__4)
                    self.state = 153
                    self.match(ASGMTParser.NUM) 
                self.state = 158
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,13,self._ctx)

            self.state = 159
            self.match(ASGMTParser.T__6)
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
            return ASGMTParser.RULE_sub

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

        localctx = ASGMTParser.SubContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_sub)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 161
            self.match(ASGMTParser.T__9)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[1] = self.mat_expr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def mat_expr_sempred(self, localctx:Mat_exprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 3)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 2)
         




