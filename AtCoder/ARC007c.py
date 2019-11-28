# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・ビット全探索
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd

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

def bit_count(i):

    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

S = input()
N = len(S)

ans = INF
# ビット全探索
for bit in range(1<<N):
    visible = [0] * (N*2)
    # テレビを付ける位置i
    for i in range(N):
        if bit & 1<<i:
            # 視聴パターンに合わせて見れる状態を更新
            for j in range(N*2):
                if S[j%N] == 'o' and i+j < N*2:
                    visible[i+j] = 1
    # 2周目の全区間が見れていればOK
    if sum(visible[N:]) == N:
        # 付けたテレビ台数の最小を取る
        ans = min(ans, bit_count(bit))
print(ans)
