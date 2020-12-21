# -*- coding: utf-8 -*-

"""
・なんかツイートで見かけたので約1振りの再挑戦。
・がしかし自力ならずTLE。。悔しい。
"""

import sys
from bisect import bisect_left, bisect_right

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

R, C, K = MAP()
N = INT()
RCNT = [0] * R
CCNT = [[0, set()] for i in range(C)]
for i in range(N):
    r, c = MAP()
    r -= 1; c -= 1
    RCNT[r] += 1
    CCNT[c][0] += 1
    CCNT[c][1].add(r)

CCNT.sort()
C2, _ = zip(*CCNT)

ans = 0
for r in range(R):
    rcnt = RCNT[r]
    # c+rがKになる
    idx1 = bisect_left(C2, K-rcnt)
    idx2 = bisect_right(C2, K-rcnt)
    cnt = idx2 - idx1
    for i in range(idx1, idx2):
        # 自分が含まれる列だと重複があるので除外
        if r in CCNT[i][1]:
            cnt -= 1
    ans += cnt
    # c+rがK+1になる
    idx1 = bisect_left(C2, K-rcnt+1)
    idx2 = bisect_right(C2, K-rcnt+1)
    cnt = idx2 - idx1
    for i in range(idx1, idx2):
        # 今度は自分を含まないと足りないので除外
        if r not in CCNT[i][1]:
            cnt -= 1
    ans += cnt
print(ans)
