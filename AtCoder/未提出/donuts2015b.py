# -*- coding: utf-8 -*-

"""
・ビット全探索
・計算量はN=16,M=50で、2^N*(N+M)=432万くらい。
　pythonAC1.7秒、pypyAC1.2秒。
・ちなみに試しにpopcountをビット見るやつにしたら、却って遅くなってpypy1.8秒。。
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

def popcount(i):
    return bin(i).count('1')

N, M = MAP()
A = LIST()
B = [0] * M
combo = [0] * M
for i in range(M):
    li = LIST()
    B[i] = li[0]
    # i番目のコンボの対象キャラの集合を取る
    for c in li[2:]:
        c -= 1
        combo[i] |= 1<<c

ans = 0
for S in range(1<<N):
    # 選択するのは9人まで
    if popcount(S) > 9:
        continue
    sm = 0
    # 基礎値
    for i in range(N):
        if S & 1<<i:
            sm += A[i]
    # コンボ
    for i in range(M):
        # コンボ対象が3人以上含まれていれば追加
        if popcount(S & combo[i]) >= 3:
            sm += B[i]
    ans = max(ans, sm)
print(ans)
