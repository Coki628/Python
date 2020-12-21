"""
・またギャグを整数BFSで殴った。これこどふぉで何回もやってる気がする。
・2で割れるなら、一番でかい約数で2にできる。
"""

import sys

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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def bfs(src):
    """ BFS(整数) """
    from collections import deque

    que = deque([src])
    dist = {}
    dist[src] = 0
    while que:
        u = que.popleft()
        if u <= 1:
            continue
        if u != 2 and u % 2 == 0:
            v = u // (u//2)
            if v not in dist:
                dist[v] = dist[u] + 1
                que.append(v)
        elif u != 3 and u % 3 == 0:
            v = u // (u//3)
            if v not in dist:
                dist[v] = dist[u] + 1
                que.append(v)
        else:
            v = u - 1
            if v not in dist:
                dist[v] = dist[u] + 1
                que.append(v)
    return dist

for _ in range(INT()):
    N = INT()

    res = bfs(N)
    ans = res[1]
    print(ans)
