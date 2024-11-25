# Generated from ASGMT.g4 by ANTLR 4.10.1
# encoding: utf-8
from antlr4 import *
import torch
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,15,93,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,1,0,1,0,1,0,1,0,1,1,1,1,3,1,27,8,1,1,1,
        5,1,30,8,1,10,1,12,1,33,9,1,1,2,1,2,1,2,3,2,38,8,2,1,2,1,2,1,3,1,
        3,1,3,3,3,45,8,3,1,3,1,3,3,3,49,8,3,1,3,3,3,52,8,3,1,4,1,4,3,4,56,
        8,4,1,5,1,5,1,5,1,5,1,5,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,7,1,7,
        1,7,3,7,74,8,7,1,7,1,7,1,7,3,7,79,8,7,5,7,81,8,7,10,7,12,7,84,9,
        7,1,7,1,7,1,8,1,8,1,9,1,9,1,9,1,9,2,31,82,0,10,0,2,4,6,8,10,12,14,
        16,18,0,1,1,0,11,12,94,0,20,1,0,0,0,2,24,1,0,0,0,4,37,1,0,0,0,6,
        51,1,0,0,0,8,55,1,0,0,0,10,57,1,0,0,0,12,62,1,0,0,0,14,70,1,0,0,
        0,16,87,1,0,0,0,18,89,1,0,0,0,20,21,3,8,4,0,21,22,5,1,0,0,22,23,
        3,2,1,0,23,1,1,0,0,0,24,31,3,4,2,0,25,27,5,2,0,0,26,25,1,0,0,0,26,
        27,1,0,0,0,27,28,1,0,0,0,28,30,3,4,2,0,29,26,1,0,0,0,30,33,1,0,0,
        0,31,32,1,0,0,0,31,29,1,0,0,0,32,3,1,0,0,0,33,31,1,0,0,0,34,35,3,
        6,3,0,35,36,5,3,0,0,36,38,1,0,0,0,37,34,1,0,0,0,37,38,1,0,0,0,38,
        39,1,0,0,0,39,40,3,6,3,0,40,5,1,0,0,0,41,52,5,12,0,0,42,52,3,18,
        9,0,43,45,3,16,8,0,44,43,1,0,0,0,44,45,1,0,0,0,45,46,1,0,0,0,46,
        52,3,8,4,0,47,49,3,16,8,0,48,47,1,0,0,0,48,49,1,0,0,0,49,50,1,0,
        0,0,50,52,3,12,6,0,51,41,1,0,0,0,51,42,1,0,0,0,51,44,1,0,0,0,51,
        48,1,0,0,0,52,7,1,0,0,0,53,56,5,11,0,0,54,56,3,10,5,0,55,53,1,0,
        0,0,55,54,1,0,0,0,56,9,1,0,0,0,57,58,5,11,0,0,58,59,5,4,0,0,59,60,
        7,0,0,0,60,61,5,5,0,0,61,11,1,0,0,0,62,63,5,6,0,0,63,64,3,14,7,0,
        64,65,5,7,0,0,65,66,3,14,7,0,66,67,5,7,0,0,67,68,3,14,7,0,68,69,
        5,8,0,0,69,13,1,0,0,0,70,73,5,4,0,0,71,74,5,12,0,0,72,74,3,18,9,
        0,73,71,1,0,0,0,73,72,1,0,0,0,74,82,1,0,0,0,75,78,5,7,0,0,76,79,
        5,12,0,0,77,79,3,18,9,0,78,76,1,0,0,0,78,77,1,0,0,0,79,81,1,0,0,
        0,80,75,1,0,0,0,81,84,1,0,0,0,82,83,1,0,0,0,82,80,1,0,0,0,83,85,
        1,0,0,0,84,82,1,0,0,0,85,86,5,5,0,0,86,15,1,0,0,0,87,88,5,9,0,0,
        88,17,1,0,0,0,89,90,5,10,0,0,90,91,5,11,0,0,91,19,1,0,0,0,10,26,
        31,37,44,48,51,55,73,78,82
    ]

class ASGMTParser ( Parser ):

    grammarFileName = "ASGMT.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'='", "'+'", "'*'", "'['", "']'", "'gm('", 
                     "','", "')'", "'-'", "'_'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "IDV", "NUM", 
                      "COMM", "WS", "DIGIT" ]

    RULE_assignment = 0
    RULE_add = 1
    RULE_add_term = 2
    RULE_term = 3
    RULE_symvars = 4
    RULE_idd = 5
    RULE_gm = 6
    RULE_list = 7
    RULE_sub = 8
    RULE_par = 9

    ruleNames =  [ "assignment", "add", "add_term", "term", "symvars", "idd", 
                   "gm", "list", "sub", "par" ]

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
    IDV=11
    NUM=12
    COMM=13
    WS=14
    DIGIT=15

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
            self.state = 20
            self.symvars()
            self.state = 21
            self.match(ASGMTParser.T__0)
            self.state = 22
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
            self.enterOuterAlt(localctx, 1)
            self.state = 24
            self.add_term()
            self.state = 31
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,1,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 26
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==ASGMTParser.T__1:
                        self.state = 25
                        self.match(ASGMTParser.T__1)


                    self.state = 28
                    self.add_term() 
                self.state = 33
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,1,self._ctx)

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
            self.state = 37
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                self.state = 34
                self.term()
                self.state = 35
                self.match(ASGMTParser.T__2)


            self.state = 39
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


        def symvars(self):
            return self.getTypedRuleContext(ASGMTParser.SymvarsContext,0)


        def sub(self):
            return self.getTypedRuleContext(ASGMTParser.SubContext,0)


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
        
        def getValue(self, data):
            if self.is_const(data):
                if not self.NUM() is None:
                    return float(self.NUM().getText())
                elif not self.symvars() is None:
                    return self.symvars().idd().getValue(data)
            else:
                raise("Calling getValue for a variable")




    def term(self):

        localctx = ASGMTParser.TermContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_term)
        self._la = 0 # Token type
        try:
            self.state = 51
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 41
                self.match(ASGMTParser.NUM)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 42
                self.par()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 44
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__8:
                    self.state = 43
                    self.sub()


                self.state = 46
                self.symvars()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 48
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==ASGMTParser.T__8:
                    self.state = 47
                    self.sub()


                self.state = 50
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
            self.state = 55
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,6,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 53
                self.match(ASGMTParser.IDV)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 54
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
                data_idx = data[self.IDV(1).getText()][0]
            return data[data_name][data_idx]




    def idd(self):

        localctx = ASGMTParser.IddContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_idd)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 57
            self.match(ASGMTParser.IDV)
            self.state = 58
            self.match(ASGMTParser.T__3)
            self.state = 59
            _la = self._input.LA(1)
            if not(_la==ASGMTParser.IDV or _la==ASGMTParser.NUM):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 60
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
            self.state = 62
            self.match(ASGMTParser.T__5)
            self.state = 63
            self.list_()
            self.state = 64
            self.match(ASGMTParser.T__6)
            self.state = 65
            self.list_()
            self.state = 66
            self.match(ASGMTParser.T__6)
            self.state = 67
            self.list_()
            self.state = 68
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
            self.state = 70
            self.match(ASGMTParser.T__3)
            self.state = 73
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [ASGMTParser.NUM]:
                self.state = 71
                self.match(ASGMTParser.NUM)
                pass
            elif token in [ASGMTParser.T__9]:
                self.state = 72
                self.par()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 82
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,9,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 75
                    self.match(ASGMTParser.T__6)
                    self.state = 78
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [ASGMTParser.NUM]:
                        self.state = 76
                        self.match(ASGMTParser.NUM)
                        pass
                    elif token in [ASGMTParser.T__9]:
                        self.state = 77
                        self.par()
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 84
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,9,self._ctx)

            self.state = 85
            self.match(ASGMTParser.T__4)
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
        self.enterRule(localctx, 16, self.RULE_sub)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 87
            self.match(ASGMTParser.T__8)
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




    def par(self):

        localctx = ASGMTParser.ParContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_par)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 89
            self.match(ASGMTParser.T__9)
            self.state = 90
            self.match(ASGMTParser.IDV)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





