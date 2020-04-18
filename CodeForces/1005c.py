# -*- coding: utf-8 -*-

"""
・2冪、式変形
・a + b = 2冪 の形にする必要があるので、式変形して b = 2冪 - a
　となるbがあるかどうかで判定していく。
・内側ループは2冪で進むので、そんなに多くならない。
・チェック時に自分を数に入れないように、処理前と処理後で自分を足し引きする。
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
MAX = 10**12

C = Counter(A)
ans = 0
for a in A:
    C[a] -= 1
    x = 1
    while x <= MAX:
        b = x - a
        if C[b] > 0:
            break
        x *= 2
    else:
        ans += 1
    C[a] += 1
print(ans)
