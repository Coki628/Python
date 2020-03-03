# -*- coding: utf-8 -*-

"""
・自力AC。やっと通った。
・2次元のDP配列を1次元のやつ2つだけにしてメモリ節約。
・これでpypyAC0.5秒。計算量は4000^2で1600万。
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

S = input()
N = len(S)
T = input()
M = len(T)

# 2次元DPをcurとnxtの2つだけにする
cur = [0] * (M+1)
ans = 0
# 後ろから埋めていく
for i in range(N-1, -1, -1):
    nxt = [0] * (M+1)
    for j in range(M-1, -1, -1):
        if S[i] == T[j]:
            nxt[j] = cur[j+1] + 1
    cur = nxt[:]
    # 古い情報は消えてしまうので、ここで最大値確認もする
    ans = max(ans, max(cur))
print(ans)
