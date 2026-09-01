# Chapter 6: the language grows its own operator library.

def unary!(value)
  if value then 0 else 1;

def unary-(value)
  0 - value;

def binary> 10 (left right)
  right < left;

# These eager logical operators deliberately match the LLVM tutorial.
def binary| 5 (left right)
  if left then 1 else if right then 1 else 0;

def binary& 6 (left right)
  if !left then 0 else !!right;

def binary= 9 (left right)
  !(left < right | left > right);

# A low-precedence sequencing operator returns its right operand.
def binary: 1 (left right)
  right;

!0;
3 > 2;
3 = 3;
1 : 2 : 42;
