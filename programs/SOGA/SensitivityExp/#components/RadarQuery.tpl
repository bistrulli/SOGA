b0 = gm([0.2,0.8], [1,0], [0,0]);

if b0 == 1 {
    b1 = 1;
} else {
    b1 = gm([0.2,0.8], [1,0], [0,0]);
} end if;

x0 = uniform([0,10], $cmp);
x1 = gm([1.], [0.], [1.4142]) + x0;

if b0 == 1 {
    x = uniform([-5, 0], $cmp) + uniform([0,5], $cmp);
    observe(x <= 0);
    y = uniform([-1, 0], $cmp) + uniform([0,1], $cmp);
    observe(y > 0);
    if gm([0.83333,0.16667],[1,0],[0,0]) > 0 {
        o0 = x0 + x;
    } else {
        o0 = x0 + y;
    } end if;
} else {
    x = uniform([-5, 0], $cmp) + uniform([0,5], $cmp);
    observe(x <= 0);
    y = uniform([-5, 0], $cmp) + uniform([0,5], $cmp);
    observe(y > 0);
    if gm([0.5,0.5],[1,0],[0,0]) > 0 {
        o0 = x + x0;
    } else {
        o0 = y + x0;
    } end if;
} end if;  

if b1 == 1 {
    x = uniform([-5, 0], $cmp) + uniform([0,5], $cmp);
    observe(x <= 0);
    y = uniform([-1, 0], $cmp) + uniform([0,1], $cmp);
    observe(y > 0);
    if gm([0.83333,0.16667],[1,0],[0,0]) > 0 {
        o1 = x + x1;
    } else {
        o1 = y + x1;
    } end if;
} else {
    x = uniform([-5, 0], $cmp) + uniform([0,5], $cmp);
    observe(x <= 0);
    y = uniform([-5, 0], $cmp) + uniform([0,5], $cmp);
    observe(y > 0);
    if gm([0.5,0.5],[1,0],[0,0]) > 0 {
        o1 = x + x1;
    } else {
        o1 = y + x1;
    } end if;
} end if;  

observe(o0 == 5);
observe(b0 == 1);



    