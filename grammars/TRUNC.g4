grammar TRUNC; 

trunc: ineq | eq | and_trunc | or_trunc;

ineq: lexpr inop const_expr;
inop: '<=' | '<' | '>' | '>=';

and_trunc: IDV inop const_expr 'and' IDV inop const_expr;
or_trunc: IDV inop const_expr 'or' IDV inop const_expr;

eq: var eqop const_expr;
eqop: '==' | '!=';

lexpr: monom (aop monom)*?;
monom: (const MUL)? var;

const_expr: const (aop pos_const)*?;
pos_const: POSNUM | par | idd;
const: num | par | idd;

num: POSNUM | SUB POSNUM;
var: IDV | idd | gm;
idd : IDV '[' (num | IDV) ']';
gm: 'gm(' list ',' list ',' list ')';
list: '[' (num | par) (',' (num | par))*? ']';

par: '_' IDV;

aop: SUM | SUB;
SUM: '+';
SUB: '-';
MUL: '*';


IDV : ALPHA (ALPHA|DIGIT)*;

POSNUM : DIGIT+ ('.' DIGIT*)?;

COMM : '/*' .*? '*/' -> skip;
WS : (' '|'\t'|'\r'|'\n') -> skip;
 
ALPHA : [a-zA-Z];
DIGIT : [0-9];