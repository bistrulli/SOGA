/* ASGMT grammar — scalar assignments and matrix assignment expressions.
   New tokens (v1): MATMUL '@', transp(X) function, X[i,j] two-arg indexing.
   ANTLR 4.10. Regenerate with:
     java -jar tools/antlr-4.10-complete.jar -Dlanguage=Python3 -visitor -listener grammars/ASGMT.g4 -o src/
*/
grammar ASGMT;

/* assignment: unified rule covering both scalar and matrix RHS.
   Preserves enterAssignment callback for existing listener in libSOGAupdate.py.
   ALTERNATIVE ORDER MATTERS: scalar `add` MUST come first. With ANTLR4 adaptive
   LL(*), when two alternatives match equally (e.g. `x = y` where `y` is bare IDV
   that fits BOTH mat_expr (via mat_atom -> IDV) AND add (via add_term -> term ->
   symvars -> IDV)), the parser picks the first-declared alternative. Putting
   `add` first ensures all M1-era scalar programs (ClickGraphPrune, etc.) continue
   to route through the scalar listener (enterAdd in libSOGAupdate.AsgmtRule).
   Matrix-only constructs (X @ Y, transp(X), matrix_gm(...)) do not match `add`
   and fall through to the matrix path automatically. M2 will add LHS-type
   routing in update_rule for true matrix assignments. */
assignment: symvars '=' add
          | symvars '=' mat_expr
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