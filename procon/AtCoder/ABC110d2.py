# -*- coding: utf-8 -*-
 
"""
参考：http://drken1215.hatenablog.com/entry/2018/09/23/224100
　　　http://tutuz.hateblo.jp/entry/2018/09/24/121248
・素因数分解と重複組み合わせ
"""
 
from collections import defaultdict
from math import sqrt, factorial
 
MOD = 10 ** 9 + 7
 
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
        # 最後までそのまま来たやつはnumが素数(ただし1^1は1^0なので数に入れない)
        if i == end and len(d) == 0 and num != 1:
            d[num] += 1
    return d
# ※このままだと最後の素数が拾えない時があるから次のやつで直した ex.)28での7とか
 
# 組み合わせの数(MOD付)
def nCr(n, r):
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial(n) % MOD
    # 分母の計算
    denominator = pow(factorial(r) * factorial(n-r), MOD-2, MOD) % MOD
    return numerator * denominator % MOD
 
N, M = map(int, input().split())

ans = 1
for k, v in fact_prime(M).items():
    # 重複組み合わせ
    # v個のkとN-1個の仕切りから、v個を並べる(v+N-1個のマスからv個を選び出す組み合わせ)
    ans = (ans * nCr(v+N-1, v)) % MOD
print(ans)
