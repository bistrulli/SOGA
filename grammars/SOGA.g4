grammar SOGA; 

progr : (data ';')*? (instr ';' | array ';')*;

data : 'data' symvars '=' list;

array: 'array[' NUM ']' IDV;

instr : assignment | conditional | prune | observe | loop;

assignment: symvars '=' (const | add | mul | exp | sin | cos) | 'skip';

const: const_term (('+'|'-') const_term)*?;
const_term: (NUM | par | idd) ('*' (NUM | idd | par))?;
add: add_term (('+'|'-') add_term)*?;
add_term: ((NUM | idd | par) '*')? vars | const_term;
mul: ((NUM | idd | par) '*')? vars '*' vars;
exp: 'exp(' symvars ')';
sin: 'sin(' symvars ')';
cos: 'cos(' symvars ')';

conditional: ifclause elseclause 'end if';

ifclause : 'if' bexpr '{' block '}';
elseclause : 'else' '{' block '}';
block : (instr ';')+;
bexpr : lexpr ('<'|'<='|'>='|'>') (NUM | idd | par) | symvars ('=='|'!=') (NUM | idd | par);
lexpr: monom (('+'|'-') monom)*?;
monom: ((NUM | idd | par) '*')? vars;

prune : 'prune(' NUM ')';

observe: 'observe(' bexpr ')';

loop : 'for' IDV 'in range(' (NUM | idd) ')' '{' block '}' 'end for';

vars: symvars | gm | uniform;
idd: IDV '[' (NUM | IDV) ']';
symvars : IDV | idd;
gm: 'gm(' list ',' list ',' list ')';
uniform: 'uniform(' list ',' NUM ')';
list: '[' (NUM | par) (',' (NUM | par))*? ']';

par: '_' IDV;

IDV : ALPHA (ALPHA|DIGIT)*;
NUM : '-'? DIGIT+ ('.' DIGIT*)?;

COMM : '/*' .*? '*/' -> skip;
WS : (' '|'\t'|'\r'|'\n') -> skip;

fragment 
ALPHA : [a-zA-Z];
DIGIT : [0-9];