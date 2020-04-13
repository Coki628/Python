# -*- coding: utf-8 -*-

"""
・自力AC
・集計して愚直にやる。
　開始位置を1〜Nまで全部試せる。
・一瞬、内側のループが重くなりそうに見えたけど、
　これは2,4,8,16..って増えてくやつなので高々logくらいしかない。
・ちょうどの位置を探すのににぶたんのlogも乗るけど、
　20万log2つ(多分)乗っても、pypyAC0.4秒。
"""

import sys
from collections import Counter
from bisect import bisect_left

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

N = INT()
A = LIST()

C = sorted(Counter(A).values())
M = len(C)

ans = 0
for x in range(1, N+1):
    idx = 0
    cnt = 0
    while 1:
        idx = bisect_left(C, x, lo=idx)
        if idx < M:
            cnt += x
            x *= 2
            idx += 1
        else:
            break
    ans = max(ans, cnt)
print(ans)
