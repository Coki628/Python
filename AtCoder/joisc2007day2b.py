# -*- coding: utf-8 -*-

"""
参考：https://tsuzu.hateblo.jp/entry/2017/02/17/014623
　　　http://algoogle.hadrori.jp/joi/sc2007fermat/
・蟻本演習2-6-6
・繰り返し二乗法
・TLE想定だけど練習のため。
・C++で定数倍改善してACした。
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
INF = float('inf')
MOD = 10 ** 9 + 7

# python標準装備だけど、練習のために自作
def pow(x, n, mod):
    # 繰り返し二乗法
    res = 1
    while n > 0:
        if n & 1:
            res = (res * x) % mod
        x = (x * x) % mod
        n >>= 1
    return res

P = INT()
N = INT()

cnts = [0] * P
for i in range(P):
    # 0..Pについて、i^N が mod P でいくつになるかを集計しておく
    cnts[pow(i, N, P)] += 1

ans = 0
for x in range(P):
    for y in range(P):
        # x,yが決まればzも定まる
        z = (x+y) % P
        ans += cnts[x] * cnts[y] * cnts[z]
print(ans)
