# Generated from ASGMT.g4 by ANTLR 4.10
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .ASGMTParser import ASGMTParser
else:
    from ASGMTParser import ASGMTParser

# This class defines a complete generic visitor for a parse tree produced by ASGMTParser.

class ASGMTVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by ASGMTParser#assignment.
    def visitAssignment(self, ctx:ASGMTParser.AssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#mat_expr.
    def visitMat_expr(self, ctx:ASGMTParser.Mat_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#mat_atom.
    def visitMat_atom(self, ctx:ASGMTParser.Mat_atomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#transp.
    def visitTransp(self, ctx:ASGMTParser.TranspContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#matrix_gm.
    def visitMatrix_gm(self, ctx:ASGMTParser.Matrix_gmContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#mlist.
    def visitMlist(self, ctx:ASGMTParser.MlistContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#mat_idd.
    def visitMat_idd(self, ctx:ASGMTParser.Mat_iddContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#add.
    def visitAdd(self, ctx:ASGMTParser.AddContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#add_term.
    def visitAdd_term(self, ctx:ASGMTParser.Add_termContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#term.
    def visitTerm(self, ctx:ASGMTParser.TermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#symvars.
    def visitSymvars(self, ctx:ASGMTParser.SymvarsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#idd.
    def visitIdd(self, ctx:ASGMTParser.IddContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#gm.
    def visitGm(self, ctx:ASGMTParser.GmContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#list.
    def visitList(self, ctx:ASGMTParser.ListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ASGMTParser#sub.
    def visitSub(self, ctx:ASGMTParser.SubContext):
        return self.visitChildren(ctx)



del ASGMTParser