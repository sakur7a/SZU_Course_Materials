/* 
 * CS:APP Data Lab 
 * 
 * <wangjinzheng 2024270223>
 * 
 * bits.c - Source file with your solutions to the Lab.
 *          This is the file you will hand in to your instructor.
 *
 * WARNING: Do not include the <stdio.h> header; it confuses the dlc
 * compiler. You can still use printf for debugging without including
 * <stdio.h>, although you might get a compiler warning. In general,
 * it's not good practice to ignore compiler warnings, but in this
 * case it's OK.  
 */

#if 0
/*
 * Instructions to Students:
 *
 * STEP 1: Read the following instructions carefully.
 */

You will provide your solution to the Data Lab by
editing the collection of functions in this source file.

INTEGER CODING RULES:
 
  Replace the "return" statement in each function with one
  or more lines of C code that implements the function. Your code 
  must conform to the following style:
 
  int Funct(arg1, arg2, ...) {
      /* brief description of how your implementation works */
      int var1 = Expr1;
      ...
      int varM = ExprM;

      varJ = ExprJ;
      ...
      varN = ExprN;
      return ExprR;
  }

  Each "Expr" is an expression using ONLY the following:
  1. Integer constants 0 through 255 (0xFF), inclusive. You are
      not allowed to use big constants such as 0xffffffff.
  2. Function arguments and local variables (no global variables).
  3. Unary integer operations ! ~
  4. Binary integer operations & ^ | + << >>
    
  Some of the problems restrict the set of allowed operators even further.
  Each "Expr" may consist of multiple operators. You are not restricted to
  one operator per line.

  You are expressly forbidden to:
  1. Use any control constructs such as if, do, while, for, switch, etc.
  2. Define or use any macros.
  3. Define any additional functions in this file.
  4. Call any functions.
  5. Use any other operations, such as &&, ||, -, or ?:
  6. Use any form of casting.
  7. Use any data type other than int.  This implies that you
     cannot use arrays, structs, or unions.

 
  You may assume that your machine:
  1. Uses 2s complement, 32-bit representations of integers.
  2. Performs right shifts arithmetically.
  3. Has unpredictable behavior when shifting an integer by more
     than the word size.

EXAMPLES OF ACCEPTABLE CODING STYLE:
  /*
   * pow2plus1 - returns 2^x + 1, where 0 <= x <= 31
   */
  int pow2plus1(int x) {
     /* exploit ability of shifts to compute powers of 2 */
     return (1 << x) + 1;
  }

  /*
   * pow2plus4 - returns 2^x + 4, where 0 <= x <= 31
   */
  int pow2plus4(int x) {
     /* exploit ability of shifts to compute powers of 2 */
     int result = (1 << x);
     result += 4;
     return result;
  }

FLOATING POINT CODING RULES

For the problems that require you to implent floating-point operations,
the coding rules are less strict.  You are allowed to use looping and
conditional control.  You are allowed to use both ints and unsigneds.
You can use arbitrary integer and unsigned constants.

You are expressly forbidden to:
  1. Define or use any macros.
  2. Define any additional functions in this file.
  3. Call any functions.
  4. Use any form of casting.
  5. Use any data type other than int or unsigned.  This means that you
     cannot use arrays, structs, or unions.
  6. Use any floating point data types, operations, or constants.


NOTES:
  1. Use the dlc (data lab checker) compiler (described in the handout) to 
     check the legality of your solutions.
  2. Each function has a maximum number of operators (! ~ & ^ | + << >>)
     that you are allowed to use for your implementation of the function. 
     The max operator count is checked by dlc. Note that '=' is not 
     counted; you may use as many of these as you want without penalty.
  3. Use the btest test harness to check your functions for correctness.
  4. Use the BDD checker to formally verify your functions
  5. The maximum number of ops for each function is given in the
     header comment for each function. If there are any inconsistencies 
     between the maximum ops in the writeup and in this file, consider
     this file the authoritative source.

/*
 * STEP 2: Modify the following functions according the coding rules.
 * 
 *   IMPORTANT. TO AVOID GRADING SURPRISES:
 *   1. Use the dlc compiler to check that your solutions conform
 *      to the coding rules.
 *   2. Use the BDD checker to formally verify that your solutions produce 
 *      the correct answers.
 */


#endif
//1
/* 
 * bitXor - x^y using only ~ and & 
 *   Example: bitXor(4, 5) = 1
 *   Legal ops: ~ &
 *   Max ops: 14
 *   Rating: 1
 */
int bitXor(int x, int y) {
  return ~(~x & ~y) & ~(x & y);
}
/* 
 * tmin - return minimum two's complement integer 
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 4
 *   Rating: 1
 */
int tmin(void) {
  return 1 << 31;
}
//2
/*
 * isTmax - returns 1 if x is the maximum, two's complement number,
 *     and 0 otherwise 
 *   Legal ops: ! ~ & ^ | +
 *   Max ops: 10
 *   Rating: 2
 */
int isTmax(int x) {
  return !((x + 1) ^ ~x) & !!(x ^ ~0);
}
/* 
 * allOddBits - return 1 if all odd-numbered bits in word set to 1
 *   Examples allOddBits(0xFFFFFFFD) = 0, allOddBits(0xAAAAAAAA) = 1
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 12
 *   Rating: 2
 */
int allOddBits(int x) {
  int mask = 0xAA | (0xAA << 8);
  mask = mask | (mask << 16);

  return !((mask & x) ^ mask); // 提取奇数位，和mask比较
}
/* 
 * negate - return -x 
 *   Example: negate(1) = -1.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 5
 *   Rating: 2
 */
int negate(int x) {
  return ~x + 1;
}
//3
/* 
 * isAsciiDigit - return 1 if 0x30 <= x <= 0x39 (ASCII codes for characters '0' to '9')
 *   Example: isAsciiDigit(0x35) = 1.
 *            isAsciiDigit(0x3a) = 0.
 *            isAsciiDigit(0x05) = 0.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 15
 *   Rating: 3
 */
int isAsciiDigit(int x) {
  return !((x + ~0x30 + 1) >> 31) & !((0x39 + ~x + 1) >> 31);
}
/* 
 * conditional - same as x ? y : z 
 *   Example: conditional(2,4,5) = 4
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 16
 *   Rating: 3
 */
int conditional(int x, int y, int z) {
  int flag = !!x;
  int mask = ~flag + 1;
  return (mask & y) | (~mask & z);
}
/* 
 * isLessOrEqual - if x <= y  then return 1, else return 0 
 *   Example: isLessOrEqual(4,5) = 1.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 24
 *   Rating: 3
 */
int isLessOrEqual(int x, int y) {
  int sign_x = (x >> 31) & 1;
  int sign_y = (y >> 31) & 1;
  int sign_diff = sign_x ^ sign_y;
  int y_minus_x_sign = ((y + (~x + 1)) >> 31) & 1;

  return (sign_diff & sign_x) | ((sign_diff ^ 1) & (y_minus_x_sign ^ 1));
}
//4
/* 
 * logicalNeg - implement the ! operator, using all of 
 *              the legal operators except !
 *   Examples: logicalNeg(3) = 0, logicalNeg(0) = 1
 *   Legal ops: ~ & ^ | + << >>
 *   Max ops: 12
 *   Rating: 4
 */
int logicalNeg(int x) {
  return ((x | (~x + 1)) >> 31) + 1; // 如果x != 0，右移后会变成全1二进制
}
/* howManyBits - return the minimum number of bits required to represent x in
 *             two's complement
 *  Examples: howManyBits(12) = 5
 *            howManyBits(298) = 10
 *            howManyBits(-5) = 4
 *            howManyBits(0)  = 1
 *            howManyBits(-1) = 1
 *            howManyBits(0x80000000) = 32
 *  Legal ops: ! ~ & ^ | + << >>
 *  Max ops: 90
 *  Rating: 4
 */
int howManyBits(int x) {
  int sign = x >> 31;
  int b16, b8, b4, b2, b1, b0;

  x = x ^ sign;

  b16 = (!!(x >> 16)) << 4;
  b8 = (!!(x >> (b16 + 8))) << 3;
  b4 = (!!(x >> (b16 + b8 + 4))) << 2;
  b2 = (!!(x >> (b16 + b8 + b4 + 2))) << 1;
  b1 = (!!(x >> (b16 + b8 + b4 + b2 + 1)));
  b0 = !!(x >> (b16 + b8 + b4 + b2 + b1));

  return b16 + b8 + b4 + b2 + b1 + b0 + 1;
}
//float
/* 
 * float_twice - Return bit-level equivalent of expression 2*f for
 *   floating point argument f.
 *   Both the argument and result are passed as unsigned int's, but
 *   they are to be interpreted as the bit-level representation of
 *   single-precision floating point values.
 *   When argument is NaN, return argument
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
unsigned float_twice(unsigned uf) {
  // 提取符号位s、8位指数exp、23位尾数frac
  unsigned s = uf >> 31;          // 符号位（1位）
  unsigned exp = (uf >> 23) & 0xFF; // 指数位（8位）
  unsigned frac = uf & 0x7FFFFF;   // 尾数位（23位）

  // 指数全1，直接返回原数
  if (exp == 0xFF) {
    return uf;
  }
  // 规格化数（指数≠0），乘以2 = 指数+1
  else if (exp != 0) {
    exp++;
    // 指数+1后变成全1 ，变为无穷大，尾数清零
    if (exp == 0xFF) {
      frac = 0;
    }
  }
  // 非规格化数（指数=0），乘以2 = 尾数左移1位
  else {
    frac <<= 1;
  }

  // 拼接所有位，返回结果
  return (s << 31) | (exp << 23) | frac;
}
/* 
 * float_i2f - Return bit-level equivalent of expression (float) x
 *   Result is returned as unsigned int, but
 *   it is to be interpreted as the bit-level representation of a
 *   single-precision floating point values.
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
unsigned float_i2f(int x) {
  unsigned sign, absx, exp, frac, tail, round;
  int shift;
  if (x == 0) {
    return 0;
  }

  sign = x & 0x80000000;
  // 先把x赋值给绝对值变量
  absx = x;
  // 如果是负数，计算补码得到绝对值
  if (sign) {
    absx = ~absx + 1;
  }

  shift = 0;
  // 循环左移，直到绝对值的最高位为1，统计移位次数
  while ((absx & 0x80000000) == 0) {
    absx = absx << 1;
    shift = shift + 1;
  }

  // 计算浮点数指数：127 + 最高位1的位置
  exp = 158 + (~shift + 1);
  // 截取尾数：去掉最高位的1，取中间23位
  frac = (absx & 0x7FFFFFFF) >> 8;
  // 取出低8位，用于舍入判断
  tail = absx & 0xFF;
  // IEEE向偶舍入，大于0x80进位，等于0x80且尾数最低位为1则进位
  round = (tail > 0x80) || ((tail == 0x80) && (frac & 1));
  // 对尾数进行舍入
  frac = frac + round;
  
  if (frac >> 23) { // 如果舍入后尾数溢出（超过23位），指数+1，尾数清零溢出位
    exp = exp + 1;
    frac = frac & 0x7FFFFF;
  }
  return sign | (exp << 23) | frac;
}
/* 
 * float_f2i - Return bit-level equivalent of expression (int) f
 *   for floating point argument f.
 *   Argument is passed as unsigned int, but
 *   it is to be interpreted as the bit-level representation of a
 *   single-precision floating point value.
 *   Anything out of range (including NaN and infinity) should return
 *   0x80000000u.
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
int float_f2i(unsigned uf) {
  // 提取符号、指数、尾数
  unsigned s = uf >> 31;
  unsigned exp = (uf >> 23) & 0xFF;
  unsigned frac = uf & 0x7FFFFF;
  int result;

  // 计算真实指数 E
  int E = exp - 127;

  // NaN / 无穷大，溢出返回0x80000000
  if (exp == 0xFF) {
    return 0x80000000u;
  }

  // E < 0 ，浮点数小于1，转int=0
  if (E < 0) {
    return 0;
  }
  // E > 31，超出int范围，溢出
  if (E > 31) {
    return 0x80000000u;
  }

  // 正常转换：加上隐含的1，移位得到整数
  // 隐含整数1，拼接尾数：1.xxxx
  result = (1 << 23) | frac;
  // 右移：还原真实数值 (2^E 对应移位)
  result = result >> (23 - E);

  // 处理符号位
  if (s == 1) {
    result = -result;
  }

  return result;
}
