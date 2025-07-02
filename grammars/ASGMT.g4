grammar ASGMT; 

assignment: symvars '=' add;

add: add_term (('+')? add_term)*? | exp | sin | cos;
add_term: (term '*')? term;

term: NUM | sub? par | sub? symvars | sub? gm;
symvars : IDV | idd;
idd : IDV '[' (NUM | IDV) ']';
gm: 'gm(' list ',' list ',' list ')';
list: '[' (NUM | par) (',' (NUM | par))*? ']';

exp: 'exp(' symvars ')';
sin: 'sin(' symvars ')';
cos: 'cos(' symvars ')';

sub: '-';

par: '_' IDV;

IDV : ALPHA (ALPHA|DIGIT)*;
NUM : ('-')? DIGIT+ ('.' DIGIT*)?;

COMM : '/*' .*? '*/' -> skip;
WS : (' '|'\t'|'\r'|'\n') -> skip;

fragment 
ALPHA : [a-zA-Z];
DIGIT : [0-9];