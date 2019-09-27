# -*- coding: utf-8 -*-

"""
・自力WA、それもサンプル以外全部。。
"""

import sys
from collections import deque
from bisect import bisect_right

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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
S = input()
M = INT()
DF = []
for _ in range(M):
    d, f = input().split()
    DF.append((d, f))
TP = {}
for _ in range(INT()):
    t, p = MAP()
    TP[t-1] = p - 1

# ランレングス圧縮(位置取得したいので持たせるのは長さじゃなくindex)
idxs, colors = deque(), deque()
cur = S[0]
idx = 0
for i in range(1, N):
    if cur != S[i]:
        idxs.append(idx)
        colors.append(cur)
        cur = S[i]
        idx = i
idxs.append(idx)
colors.append(cur)

# indexの変化を管理する
l_offset = 0
r_offset = N-1
for i in range(M):
    d, f = DF[i]
    if d == 'L':
        l_offset += 1
        if f != colors[0]:
            # 種類が1つだけの時は反対の色でも挟めないので追加
            if len(idxs) == 1:
                colors.appendleft(f)
                idxs.appendleft(-l_offset)
            # 他は挟めるので、色の区間が1つ減る
            else:
                colors.popleft()
                idxs.popleft()
                idxs[0] = -l_offset
    else:
        r_offset += 1
        if f != colors[-1]:
            if len(idxs) == 1:
                colors.append(f)
                idxs.append(r_offset)
            else:
                colors.pop()
                idxs.pop()
    # クエリがある時
    if i in TP:
        p = TP[i]
        target = p - l_offset
        # offsetでindexを合わせて、二分探索で位置取得
        idx = bisect_right(idxs, target) - 1
        print(colors[idx])
