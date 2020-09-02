"""
・自力AC！
・グラフ、SCC、DAG、メモ化再帰
・まず、閉路になるところの商品は取れないことになる。
・で、SCCして閉路になってないところだけ数えたらWA。足りなかった。
・必要だったのは、取れなくなった商品に依存性がある商品も取れなくなっていること。
　これを確認するために、閉路に入ったら即NG確定、みたいにしてメモ化再帰する。これでAC。
"""

import sys
from collections import Counter

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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def SCC(N, edges):
    """ 強連結成分分解 """

    nodes1 = [[] for i in range(N)]
    nodes2 = [[] for i in range(N)]
    for u, v in edges:
        nodes1[u].append(v)
        nodes2[v].append(u)

    T = []
    visited = [False] * N
    def rec1(cur):
        visited[cur] = True
        for nxt in nodes1[cur]:
            if not visited[nxt]:
                rec1(nxt)
        T.append(cur)

    for u in range(N):
        if not visited[u]:
            rec1(u)

    visited = [False] * N
    group = [0] * N
    grpnum = 0
    def rec2(cur):
        group[cur] = grpnum
        visited[cur] = True
        for nxt in nodes2[cur]:
            if not visited[nxt]:
                rec2(nxt)

    for u in reversed(T):
        if not visited[u]:
            rec2(u)
            grpnum += 1
    return grpnum, group

N, M = MAP()
edges = []
nodes = [[] for i in range(N)]
for i in range(M):
    g, r = MAP()
    g -= 1
    H = [h-1 for h in LIST()]
    for h in H:
        edges.append((g, h))
        nodes[g].append(h)

grpnum, group = SCC(N, edges)
C = Counter(group)
ok = [1] * N
for i, g in enumerate(group):
    if C[g] > 1:
        ok[i] = 0

memo = [-1] * N
def dfs(u):
    if memo[u] != -1:
        return memo[u]
    if not ok[u]:
        memo[u] = 0
        return 0
    res = 1
    for v in nodes[u]:
        res &= dfs(v)
    memo[u] = res
    return res

for i in range(N):
    dfs(i)
ans = sum(memo)
print(ans)
