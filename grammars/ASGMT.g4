/* ASGMT grammar — scalar assignments and matrix assignment expressions.
   New tokens (v1): MATMUL '@', transp(X) function, X[i,j] two-arg indexing.
   ANTLR 4.10. Regenerate with:
     java -jar tools/antlr-4.10-complete.jar -Dlanguage=Python3 -visitor -listener grammars/ASGMT.g4 -o src/
*/
grammar ASGMT;

/* assignment: unified rule covering both scalar and matrix RHS.
   Preserves enterAssignment callback for existing listener in libSOGAupdate.py.
   NOTE on parser permissiveness: a scalar-looking RHS like `x = y` parses as
   mat_expr (because mat_atom -> IDV matches bare identifiers). This is INTENTIONAL.
   Dispatch in libSOGAupdate.py is NOT done by ctx.add() / ctx.mat_expr() null-checks
   (which would be unreliable due to this ambiguity). Instead, routing in update_rule
   (see plan M2.4) inspects the LHS variable name and looks it up in
   dist.var_entries — matrix variables route to update_rule_matrix, scalar variables
   route to the existing scalar path. The full RHS text is recovered via
   ctx.getText() and re-parsed on the dispatched side as needed. */
assignment: symvars '=' mat_expr
          | symvars '=' add
          ;

/* mat_expr: matrix-level binary ops (@, +) and unary ops (transp).
   Left-recursive via ANTLR 4 mechanism.
   No alternative labels — preserves standard enterMat_expr callback.
   Dispatch in libSOGAupdate.py via ctx.MATMUL() / ctx.mat_expr() null-checks. */
mat_expr: mat_expr MATMUL mat_atom
        | mat_expr '+' mat_atom
        | mat_atom
        ;

mat_atom: transp
        | mat_idd
        | IDV
        | matrix_gm
        ;

/* transp(X): matrix transpose. Using function-call form to avoid T variable collision. */
transp: TRANSP '(' IDV ')';

/* matrix_gm constructor (mirrors SOGA.g4 — ASGMT parses RHS of matrix assignments) */
matrix_gm: MATRIX_GM '(' mlist ',' mlist ',' mlist ')';
mlist: '[' list (',' list)* ']';

/* mat_idd: two-argument matrix element access X[i,j] */
mat_idd: IDV '[' (NUM | IDV) ',' (NUM | IDV) ']';

add: add_term (('+')? add_term)*?;
add_term: (term '*')? term;

term: sub? NUM | sub? symvars | sub? gm;
symvars : IDV | idd;
idd : IDV '[' (NUM | IDV) ']';
gm: 'gm(' list ',' list ',' list ')';
list: '[' NUM (',' NUM)*? ']';

sub: '-';

/* Reserved keywords before IDV. */
TRANSP    : 'transp';
MATRIX_GM : 'matrix_gm';
MATMUL    : '@';

IDV : ALPHA (ALPHA|DIGIT)*;
NUM : ('-')? DIGIT+ ('.' DIGIT*)?;

COMM : '/*' .*? '*/' -> skip;
WS : (' '|'\t'|'\r'|'\n') -> skip;

fragment
ALPHA : [a-zA-Z];
DIGIT : [0-9];