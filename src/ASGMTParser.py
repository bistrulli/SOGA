# Generated from ASGMT.g4 by ANTLR 4.10.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
import torch
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,18,119,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,1,0,1,0,
        1,0,1,0,1,1,1,1,3,1,33,8,1,1,1,5,1,36,8,1,10,1,12,1,39,9,1,1,1,1,
        1,1,1,3,1,44,8,1,1,2,1,2,1,2,3,2,49,8,2,1,2,1,2,1,3,1,3,3,3,55,8,
        3,1,3,1,3,3,3,59,8,3,1,3,1,3,3,3,63,8,3,1,3,3,3,66,8,3,1,4,1,4,3,
        4,70,8,4,1,5,1,5,1,5,1,5,1,5,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,7,
        1,7,1,7,3,7,88,8,7,1,7,1,7,1,7,3,7,93,8,7,5,7,95,8,7,10,7,12,7,98,
        9,7,1,7,1,7,1,8,1,8,1,8,1,8,1,9,1,9,1,9,1,9,1,10,1,10,1,10,1,10,
        1,11,1,11,1,12,1,12,1,12,1,12,2,37,96,0,13,0,2,4,6,8,10,12,14,16,
        18,20,22,24,0,1,1,0,14,15,121,0,26,1,0,0,0,2,43,1,0,0,0,4,48,1,0,
        0,0,6,65,1,0,0,0,8,69,1,0,0,0,10,71,1,0,0,0,12,76,1,0,0,0,14,84,
        1,0,0,0,16,101,1,0,0,0,18,105,1,0,0,0,20,109,1,0,0,0,22,113,1,0,
        0,0,24,115,1,0,0,0,26,27,3,8,4,0,27,28,5,1,0,0,28,29,3,2,1,0,29,
        1,1,0,0,0,30,37,3,4,2,0,31,33,5,2,0,0,32,31,1,0,0,0,32,33,1,0,0,
        0,33,34,1,0,0,0,34,36,3,4,2,0,35,32,1,0,0,0,36,39,1,0,0,0,37,38,
        1,0,0,0,37,35,1,0,0,0,38,44,1,0,0,0,39,37,1,0,0,0,40,44,3,16,8,0,
        41,44,3,18,9,0,42,44,3,20,10,0,43,30,1,0,0,0,43,40,1,0,0,0,43,41,
        1,0,0,0,43,42,1,0,0,0,44,3,1,0,0,0,45,46,3,6,3,0,46,47,5,3,0,0,47,
        49,1,0,0,0,48,45,1,0,0,0,48,49,1,0,0,0,49,50,1,0,0,0,50,51,3,6,3,
        0,51,5,1,0,0,0,52,66,5,15,0,0,53,55,3,22,11,0,54,53,1,0,0,0,54,55,
        1,0,0,0,55,56,1,0,0,0,56,66,3,24,12,0,57,59,3,22,11,0,58,57,1,0,
        0,0,58,59,1,0,0,0,59,60,1,0,0,0,60,66,3,8,4,0,61,63,3,22,11,0,62,
        61,1,0,0,0,62,63,1,0,0,0,63,64,1,0,0,0,64,66,3,12,6,0,65,52,1,0,
        0,0,65,54,1,0,0,0,65,58,1,0,0,0,65,62,1,0,0,0,66,7,1,0,0,0,67,70,
        5,14,0,0,68,70,3,10,5,0,69,67,1,0,0,0,69,68,1,0,0,0,70,9,1,0,0,0,
        71,72,5,14,0,0,72,73,5,4,0,0,73,74,7,0,0,0,74,75,5,5,0,0,75,11,1,
        0,0,0,76,77,5,6,0,0,77,78,3,14,7,0,78,79,5,7,0,0,79,80,3,14,7,0,
        80,81,5,7,0,0,81,82,3,14,7,0,82,83,5,8,0,0,83,13,1,0,0,0,84,87,5,
        4,0,0,85,88,5,15,0,0,86,88,3,24,12,0,87,85,1,0,0,0,87,86,1,0,0,0,
        88,96,1,0,0,0,89,92,5,7,0,0,90,93,5,15,0,0,91,93,3,24,12,0,92,90,
        1,0,0,0,92,91,1,0,0,0,93,95,1,0,0,0,94,89,1,0,0,0,95,98,1,0,0,0,
        96,97,1,0,0,0,96,94,1,0,0,0,97,99,1,0,0,0,98,96,1,0,0,0,99,100,5,
        5,0,0,100,15,1,0,0,0,101,102,5,9,0,0,102,103,3,8,4,0,103,104,5,8,
        0,0,104,17,1,0,0,0,105,106,5,10,0,0,106,107,3,8,4,0,107,108,5,8,
        0,0,108,19,1,0,0,0,109,110,5,11,0,0,110,111,3,8,4,0,111,112,5,8,
        0,0,112,21,1,0,0,0,113,114,5,12,0,0,114,23,1,0,0,0,115,116,5,13,
        0,0,116,117,5,14,0,0,117,25,1,0,0,0,12,32,37,43,48,54,58,62,65,69,
        87,92,96
    ]

class ASGMTParser ( Parser ):

    grammarFileName = "ASGMT.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'='", "'+'", "'*'", "'['", "']'", "'gm('", 
                     "','", "')'", "'exp('", "'sin('", "'cos('", "'-'", 
                     "'_'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "IDV", "NUM", "COMM", "WS", 
                      "DIGIT" ]

    RULE_assignment = 0
    RULE_add = 1
    RULE_add_term = 2
    RULE_term = 3
    RULE_symvars = 4
    RULE_idd = 5
    RULE_gm = 6
    RULE_list = 7
    RULE_exp = 8
    RULE_sin = 9
    RULE_cos = 10
    RULE_sub = 11
    RULE_par = 12

    ruleNames =  [ "assignment", "add", "add_term", "term", "symvars", "idd", 
                   "gm", "list", "exp", "sin", "cos", "sub", "par" ]

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
    IDV=14
    NUM=15
    COMM=16
    WS=17
    DIGIT=18

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10.1")
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
            self.enterOuterAlt(localctx, 1)
            self.state = 26
            self.symvars()
            self.state = 27
            self.match(ASGMTParser.T__0)
            self.state = 28
            self.add()
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


        def exp(self):
            return self.getTypedRuleContext(ASGMTParser.ExpContext,0)


        def sin(self):
            return self.getTypedRuleContext(ASGMTParser.SinContext,0)


        def cos(self):
            return self.getTypedRuleContext(ASGMTParser.CosContext,0)


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
        self.enterRule(localctx, 2, self.RULE_add)
        self._la = 0 # Token type
        try:
            self.state = 43
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [ASGMTParser.T__5, ASGMTParser.T__11, ASGMTParser.T__12, ASGMTParser.IDV, ASGMTParser.NUM]:
                self.enterOuterAlt(localctx, 1)
                self.state = 30
                self.add_term()
                self.state = 37
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,1,self._ctx)
                while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt==1+1:
                        self.state = 32
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        if _la==ASGMTParser.T__1:
                            self.state = 31
                            self.match(ASGMTParser.T__1)


                        self.state = 34
                        self.add_term() 
                    self.state = 39
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,1,self._ctx)

                pass
            elif token in [ASGMTParser.T__8]:
                self.enterOuterAlt(localctx, 2)
                self.state = 40
                self.exp()
                pass
            elif token in [ASGMTParser.T__9]:
                self.enterOuterAlt(localctx, 3)
                self.state = 41
                self.sin()
                pass
            elif token in [ASGMTParser.T__10]:
                self.enterOuterAlt(localctx, 4)
                self.state = 42
                self.cos()
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
        self.enterRule(localctx, 4, self.RULE_add_term)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 48
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                self.state = 45
                self.term()
                self.state = 46
                self.match(ASGMTParser.T__2)


            self.state = 50
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

        def par(self):
            return self.getTypedRuleContext(ASGMTParser.ParContext,0)


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

        def is_var(self, data):
            """ Returns 1 if term is a variable, 0 if it's a constant """
            if not self.NUM() is None:
                return False
            if not self.par() is None:
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
    
        def is_const(self, data):
            return not self.is_var(data)

        def getValue(self, data, params_dict):
            if self.is_const(data):
                if not self.NUM() is None:
                    return float(self.NUM().getText())
                elif not self.symvars() is None:
                    return self.symvars().idd().getValue(data)
                elif not self.par() is None:
                    return self.par().getValue(params_dict)
            else:
                raise("Calling getValue for a variable")



    def term(self):

        localctx = ASGMTParser.TermContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_term)
        self._la = 0 # Token type
        try:
            self.state = 65
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 52
                self.match(ASGMTParser.NUM)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 54
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__11:
                    self.state = 53
                    self.sub()


                self.state = 56
                self.par()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 58
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__11:
                    self.state = 57
                    self.sub()


                self.state = 60
                self.symvars()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 62
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__11:
                    self.state = 61
                    self.sub()


                self.state = 64
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

        def getVar(self, data):
            if self.idd() is None:
                return self.getText()
            else:
                if self.idd().IDV(1) is None:
                    return self.getText()
                else:
                    data_idx = int(data[self.idd().IDV(1).getText()][0].item())
                return self.idd().IDV(0).getText()+'['+str(data_idx)+']'



    def symvars(self):

        localctx = ASGMTParser.SymvarsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_symvars)
        try:
            self.state = 69
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,8,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 67
                self.match(ASGMTParser.IDV)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 68
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

        def is_data(self, data):
            if self.IDV(0).getText() in data.keys():
                return True
            else:
                return False
                
        def getValue(self, data):
            data_name = self.IDV(0).getText()
            if not self.NUM() is None:
                data_idx = int(self.NUM().getText())
            elif not self.IDV(1) is None:
                data_idx = int(data[self.IDV(1).getText()][0].item())
            return data[data_name][data_idx]


    def idd(self):

        localctx = ASGMTParser.IddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_idd)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 71
            self.match(ASGMTParser.IDV)
            self.state = 72
            self.match(ASGMTParser.T__3)
            self.state = 73
            _la = self._input.LA(1)
            if not(_la==ASGMTParser.IDV or _la==ASGMTParser.NUM):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 74
            self.match(ASGMTParser.T__4)
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
        self.enterRule(localctx, 12, self.RULE_gm)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 76
            self.match(ASGMTParser.T__5)
            self.state = 77
            self.list_()
            self.state = 78
            self.match(ASGMTParser.T__6)
            self.state = 79
            self.list_()
            self.state = 80
            self.match(ASGMTParser.T__6)
            self.state = 81
            self.list_()
            self.state = 82
            self.match(ASGMTParser.T__7)
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

        def par(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ASGMTParser.ParContext)
            else:
                return self.getTypedRuleContext(ASGMTParser.ParContext,i)


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

        localctx = ASGMTParser.ListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 84
            self.match(ASGMTParser.T__3)
            self.state = 87
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [ASGMTParser.NUM]:
                self.state = 85
                self.match(ASGMTParser.NUM)
                pass
            elif token in [ASGMTParser.T__12]:
                self.state = 86
                self.par()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 96
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,11,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 89
                    self.match(ASGMTParser.T__6)
                    self.state = 92
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [ASGMTParser.NUM]:
                        self.state = 90
                        self.match(ASGMTParser.NUM)
                        pass
                    elif token in [ASGMTParser.T__12]:
                        self.state = 91
                        self.par()
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 98
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,11,self._ctx)

            self.state = 99
            self.match(ASGMTParser.T__4)
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
            return self.getTypedRuleContext(ASGMTParser.SymvarsContext,0)


        def getRuleIndex(self):
            return ASGMTParser.RULE_exp

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

        localctx = ASGMTParser.ExpContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_exp)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 101
            self.match(ASGMTParser.T__8)
            self.state = 102
            self.symvars()
            self.state = 103
            self.match(ASGMTParser.T__7)
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
            return self.getTypedRuleContext(ASGMTParser.SymvarsContext,0)


        def getRuleIndex(self):
            return ASGMTParser.RULE_sin

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

        localctx = ASGMTParser.SinContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_sin)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 105
            self.match(ASGMTParser.T__9)
            self.state = 106
            self.symvars()
            self.state = 107
            self.match(ASGMTParser.T__7)
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
            return self.getTypedRuleContext(ASGMTParser.SymvarsContext,0)


        def getRuleIndex(self):
            return ASGMTParser.RULE_cos

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

        localctx = ASGMTParser.CosContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_cos)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 109
            self.match(ASGMTParser.T__10)
            self.state = 110
            self.symvars()
            self.state = 111
            self.match(ASGMTParser.T__7)
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
        self.enterRule(localctx, 22, self.RULE_sub)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 113
            self.match(ASGMTParser.T__11)
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
            return self.getToken(ASGMTParser.IDV, 0)

        def getRuleIndex(self):
            return ASGMTParser.RULE_par

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

        localctx = ASGMTParser.ParContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_par)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 115
            self.match(ASGMTParser.T__12)
            self.state = 116
            self.match(ASGMTParser.IDV)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





