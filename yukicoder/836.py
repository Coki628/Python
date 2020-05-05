# -*- coding: utf-8 -*-

"""
・自力AC
・ちょうど最近こどふぉでやったやつと同じ感じで。(CF1342C)
・区間全体を含むところは全部まとめて足せる。
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

l, r, N = MAP()
# 半開区間にしておく
r += 1

ln = l // N
rn = r // N
lmod = l % N
rmod = r % N
segcnt = rn - ln
ans = [segcnt] * N
for i in range(N):
    if i < lmod:
        ans[i] -= 1
    if i < rmod:
        ans[i] += 1
for i in range(N): 
    print(ans[i])
