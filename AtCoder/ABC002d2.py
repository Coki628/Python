# -*- coding: utf-8 -*-

"""
・D自力AC！
・大昔にやって(きっと)全然できなかったやつ、完全に忘れてるので再挑戦。ACできて嬉しい。
・bitDP
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

def bit_cnt(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

N, M = MAP()
# XY[i] := 頂点iと繋がっている頂点の集合
XY = [0] * N
for i in range(M):
    x, y = MAP()
    XY[x-1] += 1<<(y-1)
    XY[y-1] += 1<<(x-1)
# 自分を追加
for i in range(N):
    XY[i] += 1<<i

dp = [False] * (1<<N)
# 初期化：自分は絶対繋がっているのでTrue
for i in range(N):
    dp[1<<i] = True

for bit in range(1<<N):
    # 集合bitに頂点iを追加できるか
    for i in range(N):
        # 2つの集合で&を取れば、全て含まれているか確認できる
        if (bit&XY[i]) == bit:
            # iが集合bitを全て含んでいればOKなので遷移させる
            dp[bit|(1<<i)] = dp[bit|(1<<i)] or dp[bit]

mx = 0
for bit in range(1<<N):
    if dp[bit]:
        # 有効な集合でビットが最も多く立っている所を取る
        mx = max(mx, bit_cnt(bit))
print(mx)
