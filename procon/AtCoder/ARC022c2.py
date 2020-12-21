# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc022
・別解。こちらの方針のが汎用性があるとのことで一応やっておく。
・木の直径、木DP
・確かに葉から順に何かを確定していく動きは、木の走査でよく使う印象。
・pythonAC0.9秒。再帰探索なのでやっぱりちょっと遅くなる。(ソートもしてるしね)
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
    u, v = MAP()
    u -= 1; v -= 1
    nodes[u].append(v)
    nodes[v].append(u)

# dp[i] := iを根とした時、自分とその子孫の中で一番遠い2頂点 (距離, 頂点a, 頂点b)
dp = [(-1, -1, -1) for i in range(N)]
def dfs(u, prev):
    tmp = [(0, u)]
    for v in nodes[u]:
        if v != prev:
            tmp.append(dfs(v, u))
    # 自分が葉でなければ
    if len(tmp) >= 2:
        tmp.sort(reverse=True)
        a, b = tmp[0], tmp[1]
        # 自分より子孫側で1,2番目に遠い2頂点
        dp[u] = (a[0]+b[0], a[1], b[1])
    # 現時点で1番遠い頂点を、自分-親間の距離を足して返す
    return (tmp[0][0]+1, tmp[0][1])

dfs(0, -1)
ans = (-1, -1, -1)
for i in range(N):
    if dp[i][0] > ans[0]:
        ans = dp[i]
print(ans[1]+1, ans[2]+1)
