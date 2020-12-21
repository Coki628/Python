# -*- coding: utf-8 -*-

import sys
from math import pi

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

a, b, c = MAP()

sm = a + b + c
mx = max(a, b, c)
others = sm - mx
# 一番長いやつが、他全部合わせたより大きいと、中心付近に行けない
dead = max(mx - others, 0)

# 円の面積
ans = sm**2 * pi
if dead:
    ans -= dead**2 * pi
print(ans)
