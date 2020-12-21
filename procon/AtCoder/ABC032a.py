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

lcm = lcm_base(a, b)
tmp = lcm
while True:
    if tmp >= n:
        print(tmp)
        exit()
    tmp += lcm