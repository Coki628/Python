"""
・自力AC
・グリッド
・実装おつとコーナーケース。
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

H, W = MAP()
grid = [[]] * H
blkcnt = 0
for i in range(H):
    grid[i] = list(input())
    blkcnt += grid[i].count('#')

# 黒マスなしはNG
if not blkcnt:
    NO()
    exit()

def check1(h, w):
    used = list2d(H, W, 0)
    for h1 in range(H):
        for w1 in range(W):
            if not used[h1][w1] and grid[h1][w1] == '#':
                h2 = h1 + h
                w2 = w1 + w
                if h2 < 0 or w2 < 0 or h2 >= H or w2 >= W or grid[h2][w2] == '.':
                    return False
                used[h2][w2] = 1
    return True

# 左下向きの確認もする
def check2(h, w):
    used = list2d(H, W, 0)
    for h1 in range(H):
        for w1 in range(W):
            if not used[h1][w1] and grid[h1][w1] == '#':
                h2 = h1 + h
                w2 = w1 - w
                if h2 < 0 or w2 < 0 or h2 >= H or w2 >= W or grid[h2][w2] == '.':
                    return False
                used[h2][w2] = 1
    return True

for i in range(H):
    for j in range(W):
        if i == j == 0:
            continue
        if check1(i, j) or check2(i, j):
            YES()
            exit()
NO()
