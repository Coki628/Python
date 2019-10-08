# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-11、自力ならず。。前やったやつだし悔しい。
・遷移、+じゃなくて*かなぁは思ったけど、結局正しい形は見つけ出せず。。
・まあ、木DPはまだこなした回数少ないし、しょうがない。また頑張る。
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
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    nodes[a-1].append(b-1)
    nodes[b-1].append(a-1)

dp1 = [0] * N
dp2 = [0] * N
def rec(u, prev):
    if prev == -1 or len(nodes[u]) >= 2:
        for v in nodes[u]:
            if v != prev:
                rec(v, u)
                dp1[u] += dp1[v]
                dp1[u] += dp2[v]
                dp2[u] += dp1[v]
    else:
        dp1[u] = 1
        dp2[u] = 1
    return

rec(0, -1)
print(dp1[0]+dp2[0])
