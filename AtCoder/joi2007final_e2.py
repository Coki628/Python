# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-8、自力AC！
・座標圧縮、2次元いもす、グリッドDFS
・制約(1.5秒、64M)的に100万グリッドつらい。pythonTLE、pypyMLE。
・AOJの方でAC確認。
・累積和の時回転させないでaccumulate使わない版。こっちのが遅い。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def compress(A):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(set(A))):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

W, H = MAP()
N = INT()
X = []
Y = []
XY = []
for i in range(N):
    x1, y1, x2, y2 = MAP()
    # 空いている区間を保持するために-1した場所の入れる
    X += [x1-1, x1, x2-1, x2]
    Y += [y1-1, y1, y2-1, y2]
    XY.append((x1, y1, x2, y2))

# 座標圧縮
X_zipped, _ = compress(X + [-1, 0, W])
Y_zipped, _ = compress(Y + [-1, 0, H])
for i in range(N):
    # 圧縮後の値に置き換え
    x1, y1, x2, y2 = XY[i]
    XY[i] = (X_zipped[x1], Y_zipped[y1], X_zipped[x2], Y_zipped[y2])

# ここから全部圧縮後の値でやる
H, W = len(Y_zipped), len(X_zipped)
grid = list2d(H, W, 0)
# 2次元いもす
for i in range(N):
    x1, y1, x2, y2 = XY[i]
    # 終端は半開区間ぽい感じなので+1しなくていい
    grid[y1][x1] += 1
    grid[y2][x1] -= 1
    grid[y1][x2] -= 1
    grid[y2][x2] += 1
for i in range(H):
    for j in range(W-1):
        grid[i][j+1] += grid[i][j]
for j in range(W):
    for i in range(H-1):
        grid[i+1][j] += grid[i][j]
# 枠を-1にしておく
for i in range(H):
    grid[i][0] = grid[i][W-1] = -1
for j in range(W):
    grid[0][j] = grid[H-1][j] = -1

directions = ((1,0),(-1,0),(0,1),(0,-1))
visited = list2d(H, W, 0)
def dfs(i, j):
    if visited[i][j] or grid[i][j] != 0:
        return 0
    stack = [(i, j)]
    while stack:
        h, w = stack.pop()
        visited[h][w] = 1
        for dh, dw in directions:
            h2 = h + dh
            w2 = w + dw
            if not visited[h2][w2] and grid[h2][w2] == 0:
                stack.append((h2, w2))
    return 1

cnt = 0
for i in range(1, H-1):
    for j in range(1, W-1):
        cnt += dfs(i, j)
print(cnt)
