# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力ならず。内容かなり覚えてた問題だったのに、良い解法は出なかった。
　当時のダメ方針からほぼ変化してないやつ書いて終了。。(さすがに消した)
・再帰探索応用
・食べれる長さが真ん中より手前か先かで再帰で潜る条件を場合分け。
・こういうの自分でちゃんと導けるようにしたい。
・ちなみに右側に潜る時について、右端バンズの数は、パティ数に影響ないので考慮しなくてよし。
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
MOD = 10 ** 9 + 7

N, X = MAP()

# 各レベルでの層とパティの総数を前計算
A = [0] * (N+1)
P = [0] * (N+1)
A[0] = P[0] = 1
for i in range(1, N+1):
    A[i] = A[i-1] * 2 + 3
    P[i] = P[i-1] * 2 + 1

def rec(n, x):
    if n == 0:
        # 最後の1つは食べる残数がまだ残ってれば食べる
        if x >= 1:
            return 1
        else:
            return 0
    else:
        # 食べる残数が真ん中のパティより手前
        if x <= A[n] // 2:
            # 左半分しか考えなくていい(xから左端バンズ分は引く)
            return rec(n-1, x-1)
        # 真ん中のパティ以降まで食べる
        else:
            # 左半分の総取り + 真ん中パティ + 右からいくつか(xから左半分と真ん中パティ分は引く)
            return P[n-1] + 1 + rec(n-1, x - (A[n] // 2 + 1))
print(rec(N, X))
