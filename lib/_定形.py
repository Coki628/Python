# -*- coding: utf-8 -*-

# 余りの切り上げ
(a + b - 1) / b
(a - 1) / b + 1
-(-a // b)

# 最大公約数と最小公倍数
from functools import reduce
def gcd(a, b):
    while b > 0:
        a, b = b, a%b
    return a
def lcm_base(x, y):
    return (x * y) // gcd(x, y)
def lcm_list(numbers):
    # reduce(使う関数, 足し合わせるリスト, 初期値)
    return reduce(lcm_base, numbers, initial=1)

# 素数判定用関数
import math
def is_prime_2(num):
    if num < 2:
        return False
    if num == 2 or num == 3 or num == 5:
        return True
    if num % 2 == 0 or num % 3 == 0 or num % 5 == 0:
        return False
    # 疑似素数(2でも3でも5でも割り切れない数字)で次々に割っていく
    prime = 7
    step = 4
    num_sqrt = math.sqrt(num)
    while prime <= num_sqrt:
        if num % prime == 0:
            return False
        prime += step
        step = 6 - step
    return True

# modの除算(フェルマーの小定理)
numer * pow(denomin, mod-2, mod) % mod
