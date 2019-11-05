# -*- coding: utf-8 -*-

"""
・調査用、しかし何もつかめなかった。。
・制約とXORっていう状況的に多分桁DPさせたいんだろうなぁ、、って思いながらもやーっとして終了。
・そのうちやるかぁ。。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()

for a in range(N+1):
    for b in range(a, N+1):
        u = a ^ b
        v = a + b
        # print('a: {0}, b: {1}, u: {2}, v: {3}'.format(a, b, u, v))

prev = 0
for N in range(N+1):
    cnt = set()
    for a in range(N+1):
        for b in range(a, N+1):
            u = a ^ b
            v = a + b
            if u <= N and v <= N:
                cnt.add((u, v))
    ans = len(cnt)
    print('N: {0}, ans: {1}, gap: {2}'.format(N, ans, ans-prev))
    prev = ans
