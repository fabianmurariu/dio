# Chapter 6's "kicking the tires" program, rendered at a compact size.

extern putchard(character);

def unary!(value)
  if value then 0 else 1;

def unary-(value)
  0 - value;

def binary> 10 (left right)
  right < left;

def binary| 5 (left right)
  if left then 1 else if right then 1 else 0;

def binary: 1 (left right)
  right;

def printdensity(density)
  if density > 8 then putchard(32)
  else if density > 4 then putchard(46)
  else if density > 2 then putchard(43)
  else putchard(42);

def mandelconverger(real imag iterations creal cimag)
  if iterations > 255 | (real * real + imag * imag > 4) then
    iterations
  else
    mandelconverger(
      real * real - imag * imag + creal,
      2 * real * imag + cimag,
      iterations + 1,
      creal,
      cimag
    );

def mandelconverge(real imag)
  mandelconverger(real, imag, 0, real, imag);

def mandelhelp(xmin xmax xstep ymin ymax ystep)
  for y = ymin, y < ymax, ystep in (
    (for x = xmin, x < xmax, xstep in
      printdensity(mandelconverge(x, y)))
    : putchard(10)
  );

def mandel(realstart imagstart realmag imagmag)
  mandelhelp(
    realstart,
    realstart + realmag * 30,
    realmag,
    imagstart,
    imagstart + imagmag * 16,
    imagmag
  );

mandel(-2.3, -1.3, 0.10, 0.14);
