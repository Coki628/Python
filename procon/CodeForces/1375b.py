"""
・今回唯一の自力AC。。
・グリッド、構築
・最大の状態を作っておいて、そこに持っていけるかどうかを確認すればいい。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def check(grid, ans):
    for i in range(H):
        for j in range(W):
            if grid[i][j] > ans[i][j]:
                return False
    return True

for _ in range(INT()):
    H, W = MAP()
    grid = [[]] * H
    for i in range(H):
        grid[i] = LIST()
    
    ans = list2d(H, W, 4)
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1:
                ans[i][j] -= 1
            if j == 0 or j == W-1:
                ans[i][j] -= 1
    
    if check(grid, ans):
        YES()
        for i in range(H):
            print(*ans[i])
    else:
        NO()
