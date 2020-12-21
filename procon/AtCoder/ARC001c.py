# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・8クイーン問題
・AOJでやったことあったから、ほぼ流用できた。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def build_grid(H, W, intv, _type, space=True, padding=False):
    # 入力がスペース区切りかどうか
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
    # 余白の有無
    if padding:
        offset = 1
    else:
        offset = 0
    grid = list2d(H+offset*2, W+offset*2, intv)
    for i in range(offset, H+offset):
        row = _list()
        for j in range(offset, W+offset):
            grid[i][j] = row[j-offset]
    return grid

N = 8
org = build_grid(N, N, '', str, space=False)
grid = list2d(N, N, '.')
# 横襲撃済
row = [False] * N
# 縦襲撃済
col = [False] * N
# 左下・右上襲撃済
dpos = [False] * (N*2-1)
# 左上・右下襲撃済
dneg = [False] * (N*2-1)

# クイーンを置く
def putQ(i, j):
    row[i] = True
    col[j] = True
    dpos[i+j] = True
    dneg[i-j+N-1] = True
    grid[i][j] = 'Q'

# クイーンを取り除く
def delQ(i, j):
    row[i] = False
    col[j] = False
    dpos[i+j] = False
    dneg[i-j+N-1] = False
    grid[i][j] = '.'

# 置ける状態か確認
def check(i, j):
    return not (row[i] or col[j] or dpos[i+j] or dneg[i-j+N-1])

# 初期位置を配置する
qcnt = 0
for i in range(N):
    for j in range(N):
        if org[i][j] == 'Q':
            if check(i, j):
                putQ(i, j)
                qcnt += 1
            else:
                print('No Answer')
                exit()

def rec(i, j, cnt):
    if cnt == N:
        return True
    for k in range(N):
        for l in range(N):
            if check(k, l):
                putQ(k, l)
                if rec(k, l, cnt+1):
                    return True
                else:
                    # うまくいかなかったら状態を元に戻す
                    delQ(k, l)
    return False

for i in range(N):
    for j in range(N):
        if rec(i, j, qcnt):
            [print(''.join(row)) for row in grid]
            exit()
print('No Answer')
