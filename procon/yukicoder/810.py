# -*- coding: utf-8 -*-

"""
・自力AC
・ゆきこ836をちょっと易しくした感じ。
　ほぼ同じやり方で区間内の値の数取って、全種類MとminすればOK。
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

l, r, M = MAP()
r += 1

lm = l // M
rm = r // M
segcnt = rm - lm
lmod = l % M
rmod = r % M

cnt = segcnt * M + rmod - lmod
print(min(cnt, M))
