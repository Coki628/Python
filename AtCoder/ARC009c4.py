# -*- coding: utf-8 -*-

"""
参考：http://tsutaj.hatenablog.com/entry/2016/11/02/012602
　　　https://atcoder.jp/contests/arc009/submissions/3839933
・完全順列、数え上げ、組み合わせ
・完全順列は結構有名みたいで、最初の5項くらい検索かけたら情報も出たので、それで調べても良かったな。
・今回、MODのnCrは答えが大きくなるだけでなく、入力のNからして大きいので、いつもの事前テーブルのやつだと通用しない。
　MOD付で個別計算できるやつを用意した。
・メモリ制限つらい。pypyMLE。。
・計算量はKの77万くらい。pythonTLE。。
・MOD除算を最後の1回だけにしてAC。考えてみればMOD除算の中身ってlogだから影響でかかった。
・pythonTLE → AC0.4秒。
・pypyMLE対策のためにさらに色々工夫してみる。
・DP配列を変数2つだけにしたけど、ボトルネックはこれではなかった。
・最終的に、バージョンで場合分けしたGCDインポートの文がメモリ食ってたらしい。そこ消したらpypyAC0.2秒。
"""

import sys

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
    num = den = 1
    # 分子の計算
    for i in range(n, n-r, -1):
        num *= i % MOD
        num %= MOD
    # 分母の計算
    for i in range(r, 0, -1):
        den *= i % MOD
        den %= MOD
    # logがつくため、MOD除算は最後の1回だけにする
    return num * pow(den, MOD-2, MOD) % MOD

N, K = MAP()

# ai != i となるような長さkの順列(完全順列)の通り数
def calc(k):
    # メモリ対策のため配列を変数2つに
    prev = 0
    nxt = 1
    for n in range(3, k+1):
        nxt, prev = (n-1) * (nxt + prev) % MOD, nxt
    return nxt

print(nCr(N, K, MOD) * calc(K) % MOD)
