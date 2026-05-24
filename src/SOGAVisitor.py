# Generated from SOGA.g4 by ANTLR 4.10
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .SOGAParser import SOGAParser
else:
    from SOGAParser import SOGAParser

# This class defines a complete generic visitor for a parse tree produced by SOGAParser.

class SOGAVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by SOGAParser#progr.
    def visitProgr(self, ctx:SOGAParser.ProgrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#data.
    def visitData(self, ctx:SOGAParser.DataContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#array.
    def visitArray(self, ctx:SOGAParser.ArrayContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#matrix_decl.
    def visitMatrix_decl(self, ctx:SOGAParser.Matrix_declContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#matrix_gm.
    def visitMatrix_gm(self, ctx:SOGAParser.Matrix_gmContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#mlist.
    def visitMlist(self, ctx:SOGAParser.MlistContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#instr.
    def visitInstr(self, ctx:SOGAParser.InstrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#assignment.
    def visitAssignment(self, ctx:SOGAParser.AssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#const.
    def visitConst(self, ctx:SOGAParser.ConstContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#const_term.
    def visitConst_term(self, ctx:SOGAParser.Const_termContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#add.
    def visitAdd(self, ctx:SOGAParser.AddContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#add_term.
    def visitAdd_term(self, ctx:SOGAParser.Add_termContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#mul.
    def visitMul(self, ctx:SOGAParser.MulContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#conditional.
    def visitConditional(self, ctx:SOGAParser.ConditionalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#ifclause.
    def visitIfclause(self, ctx:SOGAParser.IfclauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#elseclause.
    def visitElseclause(self, ctx:SOGAParser.ElseclauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#block.
    def visitBlock(self, ctx:SOGAParser.BlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#bexpr.
    def visitBexpr(self, ctx:SOGAParser.BexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#lexpr.
    def visitLexpr(self, ctx:SOGAParser.LexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#monom.
    def visitMonom(self, ctx:SOGAParser.MonomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#prune.
    def visitPrune(self, ctx:SOGAParser.PruneContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#observe.
    def visitObserve(self, ctx:SOGAParser.ObserveContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#loop.
    def visitLoop(self, ctx:SOGAParser.LoopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#expr.
    def visitExpr(self, ctx:SOGAParser.ExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#vars.
    def visitVars(self, ctx:SOGAParser.VarsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#transp_call.
    def visitTransp_call(self, ctx:SOGAParser.Transp_callContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#row_sum_expr.
    def visitRow_sum_expr(self, ctx:SOGAParser.Row_sum_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#col_sum_expr.
    def visitCol_sum_expr(self, ctx:SOGAParser.Col_sum_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#idd.
    def visitIdd(self, ctx:SOGAParser.IddContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#symvars.
    def visitSymvars(self, ctx:SOGAParser.SymvarsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#gm.
    def visitGm(self, ctx:SOGAParser.GmContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#uniform.
    def visitUniform(self, ctx:SOGAParser.UniformContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SOGAParser#list.
    def visitList(self, ctx:SOGAParser.ListContext):
        return self.visitChildren(ctx)



del SOGAParser