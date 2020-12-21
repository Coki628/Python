# -*- coding: utf-8 -*-

"""
・8クイーン問題
・再帰探索、バックトラック
・うまく行かなかったら状態を元に戻す。
・斜めの済確認はi+jとi-j+N-1でいける。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = 8
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

qcnt = INT()
for i in range(qcnt):
    r, c = MAP()
    putQ(r, c)

def rec(i, j, cnt):
    if cnt == 8:
        return True
    for k in range(8):
        for l in range(8):
            if check(k, l):
                putQ(k, l)
                if rec(k, l, cnt+1):
                    return True
                else:
                    # うまくいかなかったら状態を元に戻す
                    delQ(k, l)
    return False

for i in range(8):
    for j in range(8):
        if rec(i, j, qcnt):
            [print(''.join(row)) for row in grid]
            exit()
