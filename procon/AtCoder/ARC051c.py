# -*- coding: utf-8 -*-

"""
・時間かかったけど自力AC！
・logの世界で考える。ざっくりまとめて足して、残りはシミュレーション。
・X=1は数が増えないので例外処理する。
・公式解はもっとすごく簡単にやってたけど。。
"""

import sys
from math import log2
from heapq import heapify, heappop, heappush

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

N, X, Y = MAP()
A = LIST()

# 例外処理
if X == 1:
    A.sort()
    [print(a) for a in A]
    exit()

# 掛け算で数が膨大になるのでlogの世界で考える
logA = [log2(a) for a in A]
logx = log2(X)
# total: 数列全体として増やす予定の合計
total = logx * Y
# each: totalをそれぞれに均等に割り振る時の値
each = total / N
cnts = [0] * N
for i, a in enumerate(logA):
    # want: eachに達するために必要な量
    want = each - a
    if want > 0:
        # num: want増やすために必要なXをかける回数
        num = int(want // logx)
        a += logx * num
        logA[i] = a
        Y -= num
        # 何回かけたかを持っておく
        cnts[i] = num

# 残りは十分に少ないはずなのでシミュレーションする
que = [(a, i) for i, a in enumerate(logA)]
heapify(que)
for i in range(Y):
    a, i = heappop(que)
    a += logx
    cnts[i] += 1
    heappush(que, (a, i))

A2 = []
for i in range(N):
    a, i = heappop(que)
    # (Aの元index, Xをかける回数)
    A2.append((i, cnts[i]))

for i, cnt in A2:
    ans = A[i] * pow(X, cnt, MOD) % MOD
    print(ans)
