# -*- coding: utf-8 -*-

"""
・自力AC！
・木の走査、木DP
・親側の集合は自分では調べられないのでどうするか悩んだけど、全体から子側の合計と自分引けばいいと気づいた。
・計算量は10万頂点の木でpythonAC0.4秒。
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
nodes = [[] for i in range(N)]
for i in range(N-1):
    p = INT()
    nodes[i+1].append(p)
    nodes[p].append(i+1)

ans = [0] * N
def rec(u, prev):
    # 各子側の頂点集合から最大と合計を取っておく
    mx = cnt = 0
    for v in nodes[u]:
        if v != prev:
            res = rec(v, u)
            mx = max(mx ,res)
            cnt += res
    # 自分の親側の集合の大きさ
    par = N - cnt - 1
    # 子側の最大と比較
    mx = max(mx, par)
    ans[u] = mx
    # 親には子側の合計に自分を足して、自分以下の頂点集合の大きさとして返す
    return cnt + 1

rec(0, -1)
[print(a) for a in ans]
