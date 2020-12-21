# -*- coding: utf-8 -*-

from functools import reduce

N = int(input())
aN = list(map(int, input().split()))

def gcd(a, b):
    while b > 0:
        a, b = b, a%b
    return a
def lcm_base(x, y):
    return (x * y) // gcd(x, y)
def lcm_list(numbers):
    return reduce(lcm_base, numbers, 1)

def f(num, aN):
    sum1 = 0
    for i in range(len(aN)):
        sum1 += num % aN[i]
    return sum1

# リストの最小公倍数-1を関数fの引数にする
lcm = lcm_list(aN)
print(f(lcm-1, aN))