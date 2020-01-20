# -*- coding: utf-8 -*-

"""
・400点自力AC。
・実験で見つけた周期に基づいて、頑張って条件詰めた。
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

N, A, B, C, D = MAP()
# Aを起点に対称に広がるのでとりあえずAを起点とする
B -= A

# l: 有効な長さ、b: 到達できない長さ、mx: 届く最大(最小)位置
# 1回進めた時
l = D - C + 1
b = C * 2 - 1
mx = D
# さらにN-2回進めて最終的な状態にする
l += (D-C) * (N-2)
b = max(b - (D-C) * (N-2), 0)
mx *= N - 1

# 最小位置を起点とするため+mxする
B += mx
# 周期
mod = l + b
# そもそも範囲外
if B < 0 or B > mx*2:
    NO()
    exit()
# 範囲内なら、周期から到達できる位置か判定する
if B % mod < l:
    YES()
else:
    NO()
