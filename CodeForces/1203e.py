# -*- coding: utf-8 -*-

"""
・自力AC
・左から貪欲
・1283Eとほぼ同じことをする。正月がボクサーになった。
・左と自分の位置に1つずつはあるようにして、余ったら右へ。
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

C = Counter(A)

mx = max(A)
C2 = Counter()
for i in range(1, mx+1):
    if i != 1 and C2[i-1] == 0 and C[i] > 0:
        C2[i-1] += 1
        C[i] -= 1
    if C2[i] == 0 and C[i] > 0:
        C2[i] += 1
        C[i] -= 1
    C2[i+1] = C[i]
    C[i] = 0

ans = 0
for v in C2.values():
    if v > 0:
        ans += 1
print(ans)
