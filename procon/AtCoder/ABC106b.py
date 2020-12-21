# -*- coding: utf-8 -*-

N = int(input())

# 素因数分解
def factorization(n):
    def factor_sub(n, m):
        c = 0
        while n % m == 0:
            c += 1
            n /= m
        return c, n
    
    buff = []
    c, m = factor_sub(n, 2)
    if c > 0: buff.append((2, c))
    c, m = factor_sub(m, 3)
    if c > 0: buff.append((3, c))
    x = 5
    while m >= x * x:
        c, m = factor_sub(m, x)
        if c > 0: buff.append((x, c))
        if x % 6 == 5:
            x += 2
        else:
            x += 4
    if m > 1: buff.append((m, 1))
    return buff

# 約数の個数
def divisor_num(n):
    a = 1
    for _, x in factorization(n):
        a *= x + 1
    return a

if N < 105:
    print(0)
else:
    ans = 0
    for i in range(N, 104, -1):
        if i % 2 != 0:
            tmp = divisor_num(i)
            if tmp == 8:
                ans += 1
    print(ans)