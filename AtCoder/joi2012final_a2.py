# -*- coding: utf-8 -*-

"""
・ライブラリ確認、ランレングス圧縮
・速度もほぼ落ちてないぽいし、いい感じ。
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def RLE(data):
    """ ランレングス圧縮 """
    from itertools import groupby

    return [(x, len(list(grp))) for x, grp in groupby(data)]

N = INT()
A = LIST()

# 交互列との一致具合
B = [0] * N
for i in range(N):
    if A[i] == i % 2:
        B[i] = 1
    
# ランレングス圧縮
_, rle = zip(*RLE(B))

# 例外処理
if len(rle) <= 2:
    print(N)
    exit()

# 各区間iについて切り替えた時に得られる長さから、最大を取る
M = len(rle)
ans = 0
for i in range(1, M-1):
    # 両側の区間を繋げられる
    ans = max(ans, rle[i]+rle[i-1]+rle[i+1])
print(ans)
