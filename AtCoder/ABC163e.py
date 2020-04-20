# -*- coding: utf-8 -*-

"""
・自力WA。(サンプルで)
・ソートして貪欲
・あぁ、これ無理ならきっと制約的にも2次元DPなんだろうなぁ、って思って何も出なかった。。
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

N = INT()
A = LIST()

li = []
for i, a in enumerate(A):
    for j in range(N):
        li.append((a * abs(i-j), i, j))
li.sort(reverse=1)

used = [0] * N
done = [0] * N
ans = 0
for a, fr, to in li:
    if not used[to] and not done[fr]:
        used[to] = 1
        done[fr] = 1
        ans += a
print(ans)
