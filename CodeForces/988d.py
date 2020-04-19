# -*- coding: utf-8 -*-

"""
・ここまで時間内自力AC。
・冪乗のループは案外少ないをうまく使う系
・よくよく観察すると、最大を探すと言いつつ、実は要素数が3より大きいのは無理。
・自分、自分-ある2冪、自分+ある2冪、しか条件に合わないので、全要素とそこからの2冪を全部見ればOK。
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

N = INT()
A = LIST()

S = set(A)
mx = 0
ans = []
for i, a in enumerate(A):
    for j in range(32):
        tmp = [a]
        k = 2**j
        if a + k in S:
            tmp.append(a+k)
        if a - k in S:
            tmp.append(a-k)
        if len(tmp) > mx:
            mx = len(tmp)
            ans = tmp
print(len(ans))
print(*ans)
