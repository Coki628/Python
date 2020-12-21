# -*- coding: utf-8 -*-

"""
・類題攻略(nikkei2019-2-qual_bより)
・なんとか自力AC。考察、思い出すのに結構考えたけど。。
・向こうは木で、こっちは一般グラフで同じような数え上げ。
・こっちのが上位置換な感じ。
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
MOD = 10 ** 9 + 7

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
    # 親との繋ぎ方 * 兄弟との繋ぎ方
    ans *= pow((pow(2, prev, MOD)-1), v, MOD) * pow(2, v*(v-1)//2, MOD)
    ans %= MOD
    depth += 1
    prev = v
print(ans)
