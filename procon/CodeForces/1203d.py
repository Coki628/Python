# -*- coding: utf-8 -*-

"""
・なんとか自力AC
・部分列判定をO(N)でやる必要があったので、尺取っぽく動かしたら
　案の定バグって死にそうになった。なんとか直せたからよかったけど。。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

# 文字列TがSの(連続でない)部分列になっているか
def check(S, T):
    N = len(S)
    M = len(T)
    i = j = 0
    # 両方のindexを並行で進める
    while i < M:
        while j < N and T[i] != S[j]:
            j += 1
        # Sのが先に最後まで行ったらNG
        if j == N:
            return False
        j += 1
        i += 1
    return True

S = [ord(s)-97 for s in input()]
T = [ord(t)-97 for t in input()]
N = len(S)
M = len(T)

ans = 0
for i in range(N):
    for j in range(i+1, N+1):
        # 切り取る位置を全部試す
        S2 = S[:i] + S[j:]
        if check(S2, T):
            ans = max(ans, j - i)
print(ans)
