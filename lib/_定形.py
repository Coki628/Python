# -*- coding: utf-8 -*-

# 余りの切り上げ(3つとも同じ)
(a + b - 1) // b
(a - 1) // b + 1
-(-a // b)

# modの除算(フェルマーの小定理)
numer * pow(denomin, mod-2, mod) % mod

def ceil(a, b):
    return (a + b - 1) // b

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
from math import sqrt
def is_prime_2(num):
    if num < 2:
        return False
    if num == 2 or num == 3 or num == 5:
        return True
    if num % 2 == 0 or num % 3 == 0 or num % 5 == 0:
        return False
    # 疑似素数(2でも3でも割り切れない数字)で次々に割っていく
    prime = 7
    step = 4
    num_sqrt = sqrt(num)
    while prime <= num_sqrt:
        if num % prime == 0:
            return False
        prime += step
        step = 6 - step
    return True

# 約数の個数
from math import sqrt
def num_div(num):
    total = 1
    # 終点はルート切り捨て+1
    end = int(sqrt(num)) + 1
    for i in range(2, end+1):
        cnt = 0
        # 素因数分解：小さい方から割れるだけ割って指数をカウント
        while num % i == 0:
            num //= i
            cnt += 1
        # 指数+1をかけていくと約数をカウントできる
        total *= (cnt + 1)
        # 最後までそのまま来たやつは素数なので2つ
        if i == end and total == 1:
            return 2
    return total

# 約数の列挙
def num_div_set(N):
    # 1とその数はデフォで入れとく
    s = {1, N}
    for i in range(2, N//2+1):
        # 割り切れるなら、iを追加
        if N % i == 0:
            s.add(i)
    return s
# こっちのが全然速い
from math import sqrt
def num_div_set2(N):
    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て+1
    end = int(sqrt(N)) + 1
    for i in range(2, end+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s
