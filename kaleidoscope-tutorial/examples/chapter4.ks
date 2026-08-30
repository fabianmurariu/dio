# Chapter 4: backend optimization, native JIT execution, and typed host externs.
extern sin(x);
extern cos(x);

def addThree(x) 1 + 2 + x;
def unitCircle(x) sin(x) * sin(x) + cos(x) * cos(x);

addThree(4);
unitCircle(4);
