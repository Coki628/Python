# -*- coding: utf-8 -*-

def gcd(a, b):
    while b > 0:
        a, b = b, a%b
    return a
def lcm_base(x, y):
    return (x * y) // gcd(x, y)

a = int(input())
b = int(input())
n = int(input())

# 最小公倍数
lcm = lcm_base(a, b)
# n / lcm の切り上げをかける
print(lcm * (-(-n // lcm)))