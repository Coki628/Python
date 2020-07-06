"""
・速攻自力AC
・ビット全探索
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

H, W, K = MAP()
grid = [[]] * H
for i in range(H):
    grid[i] = list(input())

ans = 0
for bit1 in range(1<<H):
    for bit2 in range(1<<W):
        cnt = 0
        for i in range(H):
            for j in range(W):
                if grid[i][j] == '#' and (not bit1>>i & 1) and (not bit2>>j & 1):
                    cnt += 1
        if cnt == K:
            ans += 1
print(ans)
