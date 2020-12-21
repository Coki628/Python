# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・いもす法
・大まかな方針はすぐ出たのに、細かいとこ詰めれてなくてWA出しまくったのでいまいち。
"""

import sys
from itertools import accumulate

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

N, M = MAP()
C = [[] for i in range(M)]
for i in range(N):
    l, r, c = MAP()
    c -= 1
    C[c].append((l, r))

# 同チャンネルで連続する区間をまとめる
LR = []
for i in range(M):
    if C[i]:
        C[i].sort()
        LR.append(C[i][0])
        for j in range(1, len(C[i])):
            l1, r1 = C[i][j-1]
            l2, r2 = C[i][j]
            if r1 == l2:
                LR.pop()
                LR.append((l1, r2))
                # 3連続以上を考慮して更新
                C[i][j] = (l1, r2)
            else:
                LR.append((l2, r2))

# いもす法
MAX = 10 ** 5 + 7
imos = [0] * MAX
for l, r in LR:
    imos[l-1] += 1
    imos[r] -= 1
imos = list(accumulate(imos))
print(max(imos))
