/* TRUNC grammar — truncation/observe constraints.
   New alternatives (v1): mat_idd (X[i,j]), row_sum(IDV,NUM), col_sum(IDV,NUM).
   trace(IDV) is deferred to v2 — raises NotImplementedError in parser tests.
   ANTLR 4.10. Regenerate with:
     java -jar tools/antlr-4.10-complete.jar -Dlanguage=Python3 -visitor -listener grammars/TRUNC.g4 -o src/
*/
grammar TRUNC;

trunc: ineq | eq;

ineq: lexpr inop const;
inop: '<=' | '<' | '>' | '>=';

eq: var eqop const;
eqop: '==' | '!=';

lexpr: monom ((sum|sub) monom)*?;

/* monom: extended with matrix element access and row/col sum aggregates.
   No alternative labels — preserves enterMonom callback for existing listener.
   New alternatives (mat_idd, row_sum_expr, col_sum_expr) are dispatched in
   libSOGAtruncate.TruncRule.enterMonom via ctx.mat_idd() / ctx.row_sum_expr() /
   ctx.col_sum_expr() null-checks. Full implementation in M5. */
monom: (const '*')? var
     | (const '*')? mat_idd
     | (const '*')? row_sum_expr
     | (const '*')? col_sum_expr
     ;

/* mat_idd: two-argument matrix element access X[i,j] — mirrors ASGMT.g4 */
mat_idd: IDV '[' (NUM | IDV) ',' (NUM | IDV) ']';

/* row_sum(X, i): sum of row i elements of matrix variable X */
row_sum_expr: ROW_SUM '(' IDV ',' NUM ')';

/* col_sum(X, j): sum of column j elements of matrix variable X */
col_sum_expr: COL_SUM '(' IDV ',' NUM ')';

const: NUM | idd;
var: IDV | idd | gm;
idd : IDV '[' (NUM | IDV) ']';
gm: 'gm(' list ',' list ',' list ')';
list: '[' NUM (',' NUM)*? ']';

sum: '+';
sub: '-';

/* Reserved keywords before IDV. */
ROW_SUM : 'row_sum';
COL_SUM : 'col_sum';

IDV : ALPHA (ALPHA|DIGIT)*;
NUM : '-'? DIGIT+ ('.' DIGIT*)?;

COMM : '/*' .*? '*/' -> skip;
WS : (' '|'\t'|'\r'|'\n') -> skip;

fragment
ALPHA : [a-zA-Z];
DIGIT : [0-9];