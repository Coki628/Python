"""
・終了後自力WA
・雰囲気合っててサンプルも通るけどダメ。。
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

for _ in range(INT()):
    H, W, a, b = MAP()

    swapped = False
    if H > W:
        H, W = W, H
        a, b = b, a
        swapped = True

    if H*a != W*b:
        NO()
        continue

    if H*W < a*b:
        NO()
        continue

    grid = list2d(H, W, 0)
    for offset in range(b):
        for i in range(W):
            grid[(i+offset)%H][i%W] = 1

    if swapped:
        grid = list(zip(*grid))
        H, W = W, H

    YES()
    for i in range(H):
        print(''.join(map(str, grid[i])))
