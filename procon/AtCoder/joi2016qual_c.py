# -*- coding: utf-8 -*-

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
INF = float('inf')
MOD = 10 ** 9 + 7

H, W, D = MAP()
grid = [None] * H
for i in range(H):
    grid[i] = list(input())

def check(grid, H, W):
    cnt = 0
    for i in range(H):
        for j in range(W-D+1):
            # 範囲内に1つでも#があれがNG
            if not grid[i][j:j+D].count('#'):
                cnt += 1
    return cnt

ans = 0
ans += check(grid, H, W)
# 行列逆転
grid = list(zip(*grid))
ans += check(grid, W, H)
print(ans)
