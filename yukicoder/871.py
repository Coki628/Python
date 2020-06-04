"""
・自力AC！
・グラフに帰着、BFS、二分探索
・にぶたんでどこまで行けるか見て、未訪問の頂点はキューに入れる。
・遠い方から見ることで、近くにある訪問済の頂点を何度も確認しないようにしてる。
・想定解はもっと頭よくサクっと解いてたけど。。
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
EPS = 10 ** -10

def bfs(src):
    from collections import deque
    from bisect import bisect_left, bisect_right

    que = deque([src])
    visited = [0] * N
    visited[src] = 1
    while que:
        u = que.popleft()
        # このカエルの声が届く最大距離を左右について求める
        l = bisect_left(X, X[u]-A[u])
        r = bisect_right(X, X[u]+A[u]) - 1
        # 遠い方からチェックして、チェック済まで来たらそこより手前は見ない
        for v in range(l, u):
            if visited[v]:
                break
            visited[v] = 1
            que.append((v))
        for v in range(r, u, -1):
            if visited[v]:
                break
            visited[v] = 1
            que.append((v))
    return visited

N, K = MAP()
K -= 1
X = LIST()
A = LIST()

res = bfs(K)
ans = sum(res)
print(ans)
