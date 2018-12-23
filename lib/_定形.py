# -*- coding: utf-8 -*-

# 二番目の要素でソート
aN = [[1, 2], [3, 1]]
aN.sort(key=lambda x: x[1])

# modの除算(フェルマーの小定理)
MOD = 10 ** 9 + 7
numer = denomin = 1
numer * pow(denomin, MOD-2, MOD) % MOD

# 配列要素全部掛け(総乗)
import functools
import operator
prod = functools.partial(functools.reduce, operator.mul)
prod([1, 2, 3])
import numpy as np
np.prod([1, 2, 3])

# 余りの切り上げ(3つとも同じ)
def ceil(a, b):
    (a + b - 1) // b
    (a - 1) // b + 1
    return -(-a // b)

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
        # 1まで分解したら終了
        if num == 1:
            break
    # 最後に残ったnumの分
    if num != 1:
        total *= 2
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
# こっちのが全然速い(むしろ個数もこれにlenやる方が速いぽい)
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

# 素数列挙(エラトステネスの篩)
def eratosthenes_sieve(n):
    table = [0] * (n + 1)
    prime_list = []
    
    for i in range(2, n + 1):
        if table[i] == 0:
            prime_list.append(i)
            for j in range(i + i, n + 1, i):
                table[j] = 1
    
    return prime_list

# 素因数分解
from collections import defaultdict
from math import sqrt
def fact_prime(num):
    d = defaultdict(int)
    # 終点はルート切り捨て+1
    end = int(sqrt(num)) + 1
    for i in range(2, end+1):
        cnt = 0
        # 素因数分解：小さい方から割れるだけ割って素数をカウント
        while num % i == 0:
            num //= i
            d[i] += 1
        # 1まで分解したら終了
        if num == 1:
            break
    # 最後に残ったnumは素数(ただし1^1は1^0なので数に入れない)
    if num != 1:
        d[num] += 1
    return d

# 階乗たくさん使う時用のテーブル準備
# MAX：階乗に使う数値の最大以上まで作る
def init_factorial(MAX):
    # 階乗テーブル
    factorial = [1] * (MAX)
    factorial[0] = factorial[1] = 1
    for i in range(2, MAX):
        factorial[i] = factorial[i-1] * i % MOD
    # 逆元テーブル
    inverse = [1] * (MAX)
    # powに第三引数入れると冪乗のmod付計算を高速にやってくれる
    inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
    for i in range(MAX-2, 0, -1):
        # 最後から戻っていくこのループならMAX回powするより処理が速い
        inverse[i] = inverse[i+1] * (i+1) % MOD

# 組み合わせの数(必要な階乗と逆元のテーブルを事前に作っておく)
def nCr(n, r):
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = inverse[r] * inverse[n-r] % MOD
    return numerator * denominator % MOD
