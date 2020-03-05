# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1316/submission/72454985
・titiaさんがBFSでも通してたのでそれを参考に。
・到着地があるケースについて、毎回BFSしてたのを多点スタート1回に。→これは変わらなかった。
・無限ループケースについて、BFS呼んでたのをfor文内で直接処理に。→これでAC1.7秒。
・ちなみにDFSでやってた方に同じこと試しても速度変わらなかったから、やっぱdeque遅い説が濃厚っぽい。。
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

def bfs(grid, grid2):
    from collections import deque

    que = deque()
    for i in range(1, N+1):
        for j in range(1, N+1):
            if grid2[i][j] == '':
                if grid[i][j] == (i, j):
                    que.append(((i, j), (i, j)))
                    grid2[i][j] = 'X'

    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    chara = ('U', 'D', 'L', 'R')
    while que:
        hw, tgt = que.popleft()
        h, w = hw
        for i, (dh, dw) in enumerate(directions):
            h2 = h + dh
            w2 = w + dw
            if grid[h2][w2] == tgt and grid2[h2][w2] == '':
                grid2[h2][w2] = chara[i]
                que.append(((h2, w2), tgt))

N = INT()
grid = list2d(N+2, N+2, ())
for i in range(N):
    row = LIST()
    for j in range(N):
        grid[i+1][j+1] = (row[j*2], row[j*2+1])

grid2 = list2d(N+2, N+2, '')
bfs(grid, grid2)

directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
chara = ('D', 'U', 'R', 'L')
for h in range(1, N+1):
    for w in range(1, N+1):
        if grid2[h][w] == '' and grid[h][w] == (-1, -1):
            for i, (dh, dw) in enumerate(directions):
                h2 = h + dh
                w2 = w + dw
                if grid[h2][w2] == (-1, -1):
                    grid2[h][w] = chara[i]
                    break
            else:
                print('INVALID')
                exit()

for i in range(1, N+1):
    if grid2[i][1:-1].count(''):
        print('INVALID')
        exit()

print('VALID')
for i in range(1, N+1):
    print(''.join(grid2[i][1:-1]))
