# -*- coding: utf-8 -*-

"""
・なんとか自力AC
・MOD毎にまとめる。
・計算量は20万のNと定数倍。リストの要素すごい移動させたりしてるし、
　結構きついかと思ったけど、pypyAC0.7秒。(制約3秒)
"""

import sys
from collections import Counter

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, M = MAP()
A = LIST()

modA = [a%M for a in A]
idxs = [[] for i in range(M)]
for i, a in enumerate(modA):
    idxs[a].append(i)

K = N // M
cnt = 0
tmp = []
# 各MODの値が全部K個になるように移動させる(2周して前の方もやる)
for m in range(M*2):
    while tmp and len(idxs[m%M]) < K:
        idxs[m%M].append(tmp.pop())
    while len(idxs[m%M]) > K:
        tmp.append(idxs[m%M].pop())
    cnt += len(tmp)

# 各値がMODでいくつになったか
modans = [0] * N
for m, li in enumerate(idxs):
    for i in li:
        modans[i] = m

ans = [0] * N
for i in range(N):
    # Aからどう変化したかで場合分け
    if modA[i] < modans[i]:
        ans[i] = A[i] + (modans[i] - modA[i])
    elif modA[i] > modans[i]:
        ans[i] = A[i] + (modans[i]+M - modA[i])
    else:
        ans[i] = A[i]
print(cnt)
print(*ans)
