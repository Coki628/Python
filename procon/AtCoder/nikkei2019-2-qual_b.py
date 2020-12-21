# -*- coding: utf-8 -*-

"""
・300点自力AC
・木の繋ぎ方の数え上げ
・前に似たようなやつやったことあった気がしたのに、時間かかった。。
"""

import sys
from collections import Counter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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
MOD = 998244353

N = INT()
A = LIST()

if A[0] != 0:
    print(0)
    exit()

C = Counter(A)
if C[0] > 1:
    print(0)
    exit()

depth = 0
ans = 1
prev = 1
for k, v in sorted(C.items()):
    if k != depth:
        print(0)
        exit()
    # 各深さで、親の頂点数^今回の頂点数 通りの繋ぎ方がある
    ans *= pow(prev, v, MOD)
    ans %= MOD
    depth += 1
    prev = v
print(ans)
