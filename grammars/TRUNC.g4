grammar TRUNC; 

trunc: ineq | eq | and_trunc | or_trunc;

ineq: lexpr inop const;
inop: '<=' | '<' | '>' | '>=';

and_trunc: IDV inop const 'and' IDV inop const;
or_trunc: IDV inop const 'or' IDV inop const;

eq: var eqop const;
eqop: '==' | '!=';

lexpr: monom ((sum|sub) monom)*?;
monom: (const '*')? var;

const: NUM | par | idd;
var: IDV | idd | gm;
idd : IDV '[' (NUM | IDV) ']';
gm: 'gm(' list ',' list ',' list ')';
list: '[' (NUM | par) (',' (NUM | par))*? ']';

sum: '+';
sub: '-';

par: '_' IDV;

IDV : ALPHA (ALPHA|DIGIT)*;
NUM : '-'? DIGIT+ ('.' DIGIT*)?; 

COMM : '/*' .*? '*/' -> skip;
WS : (' '|'\t'|'\r'|'\n') -> skip;
 
ALPHA : [a-zA-Z];
DIGIT : [0-9];