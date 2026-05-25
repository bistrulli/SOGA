/* SOGA grammar — row-by-row mlist convention for matrix literals.
   New keywords (v1.1): matrix, matrix_gm, transp, row_sum, col_sum.
   New keywords (v1.2): matrix_gm_full (full-covariance constructor with auto-Kronecker detection).
   New ops (v1.1): @ (matmul).
   New syntactic forms (v1.1): 2-arg idd `IDV[i,j]`, 2D data via mlist.
   ANTLR 4.10. Regenerate with:
     java -jar tools/antlr-4.10-complete.jar -Dlanguage=Python3 -visitor -listener grammars/SOGA.g4 -o src/
   IMPORTANT: MATRIX_GM_FULL must appear BEFORE MATRIX_GM in the lexer section (longest-match rule).
*/
grammar SOGA;

progr : (data ';')*? (instr ';' | array ';' | matrix_decl ';')*;

/* F1: data accepts both flat list (scalar/array) and mlist (2D matrix data). */
data : 'data' symvars '=' (list | mlist);

array: 'array[' NUM ']' IDV;

/* matrix_decl: declares a matrix variable of shape [m][n].
   Example: matrix[4][4] Xinput; */
matrix_decl: MATRIX '[' NUM ']' '[' NUM ']' IDV;

/* matrix_gm: matrix Gaussian mixture constructor (Kronecker form).
   Arguments are three mlists: mean M (m x n), row factor U (m x m), col factor V (n x n).
   mlist is a list of lists — row-by-row, numeric literals only. */
matrix_gm: MATRIX_GM '(' mlist ',' mlist ',' mlist ')';

/* matrix_gm_full: general-covariance matrix Gaussian constructor with auto-Kronecker detection.
   Arguments are two mlists: mean M (m x n), full covariance Sigma (mn x mn).
   At runtime, SOGA tests whether Sigma is Kronecker-separable via Van Loan-Pitsianis rank-1 SVD;
   if the residual < SOGA_KRON_STRICT (default 1e-8), stores as (U, V) Kronecker factors;
   otherwise stores as dense sentinel (None, Sigma). */
matrix_gm_full: MATRIX_GM_FULL '(' mlist ',' mlist ')';

/* mlist: row-by-row matrix literal. Each inner list is one row. */
mlist: '[' list (',' list)* ']';

instr : assignment | conditional | prune | observe | loop;

assignment: symvars '=' (const | add | mul) | 'skip';

const: const_term (('+'|'-') const_term)*?;
const_term: (NUM | idd) ('*' (NUM | idd))?;
/* F2/matmul: '@' joins the +/- operators in `add`. Dispatcher in libMatrixUpdate
   inspects the body for '@' and routes to MATMUL_RAW. */
add: add_term (('+'|'-'|'@') add_term)*?;
add_term: ((NUM | idd) '*')? vars | const_term;
mul: ((NUM | idd) '*')? vars '*' vars;

conditional: ifclause elseclause 'end if';

ifclause : 'if' bexpr '{' block '}';
elseclause : 'else' '{' block '}';
block : (instr ';')+;
bexpr : lexpr ('<'|'<='|'>='|'>') (NUM | idd) | symvars ('=='|'!=') (NUM | idd);
lexpr: monom (('+'|'-') monom)*?;
monom: ((NUM | idd) '*')? vars;

prune : 'prune(' NUM ')';

observe: 'observe(' bexpr ')';

loop : 'for' IDV 'in range(' (NUM | idd) ')' '{' block '}' 'end for';

expr : lexpr | (NUM '*')? vars '*' vars | vars '^2' ;

/* F2/F4: vars now include transp_call, row_sum_expr, col_sum_expr.
   These let SOGA's top-level grammar capture the full text of matrix-style
   expressions so getText() reproduces them faithfully for downstream
   dispatchers in libMatrixUpdate / libMatrixTruncate. */
vars: symvars | gm | uniform | matrix_gm | matrix_gm_full | transp_call | row_sum_expr | col_sum_expr;
transp_call: TRANSP '(' IDV ')';
row_sum_expr: ROW_SUM '(' IDV ',' NUM ')';
col_sum_expr: COL_SUM '(' IDV ',' NUM ')';

/* F3: idd now accepts an optional 2nd index for matrix element access X[i,j].
   The 1-arg form X[i] (existing scalar array indexing) remains valid. */
idd: IDV '[' (NUM | IDV) (',' (NUM | IDV))? ']';

symvars : IDV | idd;
gm: 'gm(' list ',' list ',' list ')';
uniform: 'uniform(' list ',' NUM ')';
list: '[' NUM (',' NUM)*? ']';

/* Reserved keywords must appear BEFORE IDV to prevent shadowing.
   CRITICAL: MATRIX_GM_FULL must appear BEFORE MATRIX_GM — ANTLR longest-match
   would otherwise tokenize 'matrix_gm_full' as MATRIX_GM + IDV('_full'). */
MATRIX         : 'matrix';
MATRIX_GM_FULL : 'matrix_gm_full';
MATRIX_GM      : 'matrix_gm';
TRANSP         : 'transp';
ROW_SUM        : 'row_sum';
COL_SUM        : 'col_sum';

IDV : ALPHA (ALPHA|DIGIT)*;
NUM : '-'? DIGIT+ ('.' DIGIT*)?;

COMM : '/*' .*? '*/' -> skip;
WS : (' '|'\t'|'\r'|'\n') -> skip;

fragment
ALPHA : [a-zA-Z];
DIGIT : [0-9];
