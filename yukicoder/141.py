"""
・割とさくっと自力AC
・グラフに帰着、ダイクストラ、逆操作
・この前のAGC044Aみたいな感じにやる。実際にはもっと数学っぽく答えは一意に求まるみたいだけど、
　これでも問題なくpythonAC0.02秒。
"""

import sys
from math import gcd

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

def dijkstra(src):
    from heapq import heappush, heappop
    from collections import defaultdict

    que = [(0, *src)]
    dist = defaultdict(lambda: INF)
    dist[src] = 0
    while que:
        cost, a, b = heappop(que)
        # 逆数を取る操作
        if dist[(b, a)] > cost + 1:
            dist[(b, a)] = cost + 1
            heappush(que, (cost+1, b, a))
        # -1する操作
        if a > b:
            if b == 1:
                if dist[(1, 1)] > cost + a - 1:
                    dist[(1, 1)] = cost + a - 1
            else:
                d, m = divmod(a, b)
                if dist[(m, b)] > cost + d:
                    dist[(m, b)] = cost + d
                    heappush(que, (cost+d, m, b))
    return dist

M, N = MAP()

# とりあえず互いに素にしておく
g = gcd(M, N)
M, N = M//g, N//g
res = dijkstra((M, N))
ans = res[(1, 1)]
print(ans)
