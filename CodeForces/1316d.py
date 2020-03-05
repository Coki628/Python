# -*- coding: utf-8 -*-

"""
・自力ならず。。色々と方針も微妙にズレててダメだった。
・実装がんばれ系、グリッド
・到着地があるケースは、それぞれのマスから到着地を目指そうとしてWA。
・正解は、逆に到着地から、自分に向かってくるマスを全部辿る。
　ちょうどDAGの終端から、逆辺で自分に向かってくる頂点全部辿るみたいな感じ。
　全部チェックした後、到達できてない場所があったらNGにする。
・無限ループケースは、頑張ってサイクル作れるか判定しようとして重実装地獄。
・正解は、実は周囲4マス見て、同じ無限ループ候補があれば適当にそこに向かわせる、でOK。
　隣り合うやつが、お互いにどこかしらに向かい合うので、最終的にどこかの2マスに収束する。
　四方どこにも向かう場所がなかった場合だけNGにする。
・計算量は100万マスグリッドで、BFS(deque)だとTLE、DFS(list)だとAC1.4秒。
・やっぱりこどふぉ環境のdequeは遅いっぽいのだろうか。
"""

import sys
from collections import Counter

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def dfs(grid, grid2, src):

    h, w = src
    tgt = grid[h][w]
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    # 自分に向かってくるようにしたいので、進む方向と逆向き
    chara = ('U', 'D', 'L', 'R')
    grid2[h][w] = 'X'
    stack = [(h, w)]
    while stack:
        h, w = stack.pop()
        for i, (dh, dw) in enumerate(directions):
            h2 = h + dh
            w2 = w + dw
            # tgtに向かってくる予定のマスを全て辿っていく
            if grid[h2][w2] == tgt and grid2[h2][w2] == '':
                grid2[h2][w2] = chara[i]
                stack.append((h2, w2))

def dfs2(grid, grid2, src):

    h, w = src
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    # ここは自分から向かう方向にしたいので同じ向き
    chara = ('D', 'U', 'R', 'L')
    stack = [(h, w)]
    while stack:
        h, w = stack.pop()
        for i, (dh, dw) in enumerate(directions):
            h2 = h + dh
            w2 = w + dw
            # 無限ループ予定のマスがあったらそこに向かわせる
            if grid[h2][w2] == (-1, -1):
                grid2[h][w] = chara[i]
                break
        else:
            # このマスから四方を見てどこへも行けなかったらNG
            return False
    return True

N = INT()
grid = list2d(N+2, N+2, ())
for i in range(N):
    row = LIST()
    for j in range(N):
        grid[i+1][j+1] = (row[j*2], row[j*2+1])

grid2 = list2d(N+2, N+2, '')
for i in range(1, N+1):
    for j in range(1, N+1):
        if grid2[i][j] == '':
            if grid[i][j] == (i, j):
                dfs(grid, grid2, (i, j))
            elif grid[i][j] == (-1, -1):
                if not dfs2(grid, grid2, (i, j)):
                    print('INVALID')
                    exit()

for i in range(1, N+1):
    if grid2[i][1:-1].count(''):
        print('INVALID')
        exit()

print('VALID')
for i in range(1, N+1):
    print(''.join(grid2[i][1:-1]))
