# Chapter 5: value-producing conditionals and post-tested for loops.
extern putchard(char);

def fib(x)
  if x < 3 then
    1
  else
    fib(x - 1) + fib(x - 2);

def printstar(n)
  for i = 1, i < n, 1 in
    putchard(42);

fib(10);
printstar(5);
