# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-7
・半分全列挙
・前の見ないで書いた。途中色々バグらせたｗ
・残念ながら10秒あってもpythonTLE。。
　手元実行だと10秒もかかってないぽいんだけどなー。
"""

import sys
from itertools import combinations
from operator import itemgetter
from bisect import bisect_right

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

N, W = MAP()

VW = []
for i in range(N):
    v, w = MAP()
    VW.append((v, w))

# こっちから1つも選ばない用に番兵を入れる
VW1 = VW[:N//2] + [(0, 0)]
VW2 = VW[N//2:]

comb2 = []
for i in range(1, N+1):
    for comb in combinations(VW2, i):
        V2, W2 = zip(*comb)
        comb2.append((sum(V2), sum(W2)))

# こっちから1つも選ばない用に番兵を入れる
comb2 += [(0, 0)]
comb2.sort(key=itemgetter(0), reverse=True)
comb2.sort(key=itemgetter(1))
# 重さの小さい方から価値を最大化しておく
for i in range(1, len(comb2)):
    comb2[i] = (max(comb2[i][0], comb2[i-1][0]), comb2[i][1])

_, W2 = zip(*comb2)
ans = 0
for i in range(1, N+1):
    for comb1 in combinations(VW1, i):
        V1, W1 = zip(*comb1)
        v, w = sum(V1), sum(W1)
        idx = bisect_right(W2, W-w) - 1
        # そもそも片側だけで重さオーバー
        if idx == -1:
            continue
        ans = max(ans, v + comb2[idx][0])
print(ans)
