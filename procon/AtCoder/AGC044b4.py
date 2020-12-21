"""
参考：https://atcoder.jp/contests/agc044/submissions/13540570
・キューをスタックにしてみた版。ほぼ変化なし。
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

def dfs(src):
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    stack = []
    h, w = src
    stack.append((dist[h][w], h, w))
    while stack:
        cost, h, w = stack.pop()
        for dh, dw in directions:
            h2 = h + dh
            w2 = w + dw
            if h2 < 0 or w2 < 0 or h2 >= N or w2 >= N:
                continue
            if dist[h2][w2] > cost:
                dist[h2][w2] = cost
                stack.append((cost+seated[h2][w2], h2, w2))

N = INT()
P = [p-1 for p in LIST()]

dist = list2d(N, N, 0)
for i in range(N):
    for j in range(N):
        dist[i][j] = min(i, N-i-1, j, N-j-1)

ans = 0
seated = list2d(N, N, 1)
for p in P:
    h, w = p//N, p%N
    ans += dist[h][w]
    dfs((h, w))
    seated[h][w] = 0
print(ans)
