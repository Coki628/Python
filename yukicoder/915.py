"""
・なんとか自力AC
・グラフに帰着、整数BFS
・始点から掛け算してくより終点から割り算してく方が見通しがよくなるやつ。
・ガチャガチャ頑張って通したけど、これはその後テストケース見ておかしいとこ直したやつ。
・だいたい方針はよかったけど、引き算した時頂点が負数になってバグる時があった。
　そういうのはちゃんと気付いて確認しないとなー。。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def bfs(src):
    from collections import deque

    que = deque([src])
    dist = {}
    dist[src] = 0
    while que:
        u = que.popleft()
        # 割り切れるなら割った方がいい
        if u % c == 0:
            v = u // c
            if v not in dist:
                dist[v] = dist[u] + b
                que.append(v)
        else:
            # 割り切れないなら割り切れるとこまで引く
            v = u // c * c
            if v not in dist:
                dist[v] = dist[u] + b
                que.append(v)
            # 最後だけ、そのまま引き算頑張る方がいい時がある(ここで負数になってバグってた)
            if u < c*2 and u - (c-1) >= 0:
                v = u - (c-1)
                if v not in dist:
                    dist[v] = dist[u] + b
                    que.append(v)
    return dist

for _ in range(INT()):
    a, b, c = MAP()

    # cが1だとどうやっても前に進めない
    if c == 1:
        print(-1)
        continue

    res = bfs(a)
    if 0 in res:
        print(res[0])
    else:
        print(-1)
