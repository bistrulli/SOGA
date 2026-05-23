/* SOGA grammar — row-by-row mlist convention for matrix literals.
   New keywords (v1): matrix, matrix_gm.
   (transp lives in ASGMT.g4; row_sum, col_sum live in TRUNC.g4.)
   ANTLR 4.10. Regenerate with:
     java -jar tools/antlr-4.10-complete.jar -Dlanguage=Python3 -visitor -listener grammars/SOGA.g4 -o src/
*/
grammar SOGA;

progr : (data ';')*? (instr ';' | array ';' | matrix_decl ';')*;

data : 'data' symvars '=' list;

array: 'array[' NUM ']' IDV;

/* matrix_decl: declares a matrix variable of shape [m][n].
   Example: matrix[4][4] X_input; */
matrix_decl: MATRIX '[' NUM ']' '[' NUM ']' IDV;

/* matrix_gm: matrix Gaussian mixture constructor.
   Arguments are three mlists: mean M (m x n), row factor U (m x m), col factor V (n x n).
   mlist is a list of lists — row-by-row, numeric literals only. */
matrix_gm: MATRIX_GM '(' mlist ',' mlist ',' mlist ')';

/* mlist: row-by-row matrix literal. Each inner list is one row. */
mlist: '[' list (',' list)* ']';

instr : assignment | conditional | prune | observe | loop;

assignment: symvars '=' (const | add | mul) | 'skip';

const: const_term (('+'|'-') const_term)*?;
const_term: (NUM | idd) ('*' (NUM | idd))?;
add: add_term (('+'|'-') add_term)*?;
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

vars: symvars | gm | uniform | matrix_gm;
idd: IDV '[' (NUM | IDV) ']';
symvars : IDV | idd;
gm: 'gm(' list ',' list ',' list ')';
uniform: 'uniform(' list ',' NUM ')';
list: '[' NUM (',' NUM)*? ']';

/* Reserved keywords must appear BEFORE IDV to prevent shadowing. */
MATRIX    : 'matrix';
MATRIX_GM : 'matrix_gm';

IDV : ALPHA (ALPHA|DIGIT)*;
NUM : '-'? DIGIT+ ('.' DIGIT*)?;

COMM : '/*' .*? '*/' -> skip;
WS : (' '|'\t'|'\r'|'\n') -> skip;

fragment
ALPHA : [a-zA-Z];
DIGIT : [0-9];