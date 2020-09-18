"""
・強連結成分分解、トポロジカルソート
・実はグループ番号がちゃんとトポソ順に並んでるので、その番号に従っていけばトポソもOK。
・50万頂点、50万辺は再帰間に合うかなーって心配だったけど、無事AC。
　pypyAC2.06秒、pythonAC3.48秒。(制約5秒)
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
for i in range(M):
    a, b = MAP()
    edges.append((a, b))

grpnum, group = SCC(N, edges)
ans = [[] for i in range(grpnum)]
for i in range(N):
    ans[group[i]].append(i)

print(grpnum)
for i in range(grpnum):
    ans[i].insert(0, len(ans[i]))
    print(*ans[i])
