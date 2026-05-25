# Generated from grammars/ASGMT.g4 by ANTLR 4.10
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .ASGMTParser import ASGMTParser
else:
    from ASGMTParser import ASGMTParser

# This class defines a complete listener for a parse tree produced by ASGMTParser.
class ASGMTListener(ParseTreeListener):

    # Enter a parse tree produced by ASGMTParser#assignment.
    def enterAssignment(self, ctx:ASGMTParser.AssignmentContext):
        pass

    # Exit a parse tree produced by ASGMTParser#assignment.
    def exitAssignment(self, ctx:ASGMTParser.AssignmentContext):
        pass


    # Enter a parse tree produced by ASGMTParser#mat_expr.
    def enterMat_expr(self, ctx:ASGMTParser.Mat_exprContext):
        pass

    # Exit a parse tree produced by ASGMTParser#mat_expr.
    def exitMat_expr(self, ctx:ASGMTParser.Mat_exprContext):
        pass


    # Enter a parse tree produced by ASGMTParser#mat_atom.
    def enterMat_atom(self, ctx:ASGMTParser.Mat_atomContext):
        pass

    # Exit a parse tree produced by ASGMTParser#mat_atom.
    def exitMat_atom(self, ctx:ASGMTParser.Mat_atomContext):
        pass


    # Enter a parse tree produced by ASGMTParser#transp.
    def enterTransp(self, ctx:ASGMTParser.TranspContext):
        pass

    # Exit a parse tree produced by ASGMTParser#transp.
    def exitTransp(self, ctx:ASGMTParser.TranspContext):
        pass


    # Enter a parse tree produced by ASGMTParser#matrix_gm.
    def enterMatrix_gm(self, ctx:ASGMTParser.Matrix_gmContext):
        pass

    # Exit a parse tree produced by ASGMTParser#matrix_gm.
    def exitMatrix_gm(self, ctx:ASGMTParser.Matrix_gmContext):
        pass


    # Enter a parse tree produced by ASGMTParser#matrix_gm_full.
    def enterMatrix_gm_full(self, ctx:ASGMTParser.Matrix_gm_fullContext):
        pass

    # Exit a parse tree produced by ASGMTParser#matrix_gm_full.
    def exitMatrix_gm_full(self, ctx:ASGMTParser.Matrix_gm_fullContext):
        pass


    # Enter a parse tree produced by ASGMTParser#mlist.
    def enterMlist(self, ctx:ASGMTParser.MlistContext):
        pass

    # Exit a parse tree produced by ASGMTParser#mlist.
    def exitMlist(self, ctx:ASGMTParser.MlistContext):
        pass


    # Enter a parse tree produced by ASGMTParser#mat_idd.
    def enterMat_idd(self, ctx:ASGMTParser.Mat_iddContext):
        pass

    # Exit a parse tree produced by ASGMTParser#mat_idd.
    def exitMat_idd(self, ctx:ASGMTParser.Mat_iddContext):
        pass


    # Enter a parse tree produced by ASGMTParser#add.
    def enterAdd(self, ctx:ASGMTParser.AddContext):
        pass

    # Exit a parse tree produced by ASGMTParser#add.
    def exitAdd(self, ctx:ASGMTParser.AddContext):
        pass


    # Enter a parse tree produced by ASGMTParser#add_term.
    def enterAdd_term(self, ctx:ASGMTParser.Add_termContext):
        pass

    # Exit a parse tree produced by ASGMTParser#add_term.
    def exitAdd_term(self, ctx:ASGMTParser.Add_termContext):
        pass


    # Enter a parse tree produced by ASGMTParser#term.
    def enterTerm(self, ctx:ASGMTParser.TermContext):
        pass

    # Exit a parse tree produced by ASGMTParser#term.
    def exitTerm(self, ctx:ASGMTParser.TermContext):
        pass


    # Enter a parse tree produced by ASGMTParser#symvars.
    def enterSymvars(self, ctx:ASGMTParser.SymvarsContext):
        pass

    # Exit a parse tree produced by ASGMTParser#symvars.
    def exitSymvars(self, ctx:ASGMTParser.SymvarsContext):
        pass


    # Enter a parse tree produced by ASGMTParser#idd.
    def enterIdd(self, ctx:ASGMTParser.IddContext):
        pass

    # Exit a parse tree produced by ASGMTParser#idd.
    def exitIdd(self, ctx:ASGMTParser.IddContext):
        pass


    # Enter a parse tree produced by ASGMTParser#gm.
    def enterGm(self, ctx:ASGMTParser.GmContext):
        pass

    # Exit a parse tree produced by ASGMTParser#gm.
    def exitGm(self, ctx:ASGMTParser.GmContext):
        pass


    # Enter a parse tree produced by ASGMTParser#list.
    def enterList(self, ctx:ASGMTParser.ListContext):
        pass

    # Exit a parse tree produced by ASGMTParser#list.
    def exitList(self, ctx:ASGMTParser.ListContext):
        pass


    # Enter a parse tree produced by ASGMTParser#sub.
    def enterSub(self, ctx:ASGMTParser.SubContext):
        pass

    # Exit a parse tree produced by ASGMTParser#sub.
    def exitSub(self, ctx:ASGMTParser.SubContext):
        pass



del ASGMTParser