# -*- coding: utf-8 -*-

"""
参考：http://tsutaj.hatenablog.com/entry/2016/11/02/012602
・攪乱順列、数え上げ、組み合わせ
・攪乱順列は結構有名みたいで、最初の5項くらい検索かけたら情報も出たので、それで調べても良かったな。。
・今回、MODのnCrは答えが大きくなるだけでなく、入力のNからして大きいので、いつもの事前テーブルのやつだと通用しない。
　MOD付で個別計算できるやつを用意した。
・メモリ制限つらい。pypyMLE。。
・計算量はKの77万くらい。pythonTLE。。
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 1777777777

def nCr(n, r, MOD):
    """ 組み合わせの数(大きいnに対して定数回だけやる用) """

    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    res = 1
    # 分子の計算
    for i in range(n, n-r, -1):
        res *= i % MOD
        res %= MOD
    # 分母の計算
    for i in range(r, 0, -1):
        # MOD除算
        res *= pow(i, MOD-2, MOD)
        res %= MOD
    return res

N, K = MAP()

# ai != i となるような長さkの順列(攪乱順列)の通り数
def calc(k):
    # dp[k] := 長さkの攪乱順列の通り数
    dp = [0] * (k+1)
    dp[2] = 1
    for n in range(3, k+1):
        dp[n] = (n-1) * (dp[n-1] + dp[n-2]) % MOD
    return dp[k]

print(nCr(N, K, MOD) * calc(K) % MOD)
