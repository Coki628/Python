# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/joi/2007/2008-ho-prob_and_sol/2008-ho-review.pdf
・公式解
・開始位置をずらす、を全箇所試す、をSとT両側から見る。
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

# Sをi文字ずらしてTとの一致を見る
ans = 0
for i in range(N):
    match = [0] * M
    for j in range(M):
        if i+j >= N:
            break
        if S[i+j] == T[j]:
            match[j] = 1
    # 一致が連続する長さの最大を取る
    cnt = 0
    for ok in match:
        if ok:
            cnt += 1
        else:
            cnt = 0
        ans = max(ans, cnt)
# Tをi文字ずらしてSとの一致を見る
for i in range(M):
    match = [0] * N
    for j in range(N):
        if i+j >= M:
            break
        if T[i+j] == S[j]:
            match[j] = 1
    cnt = 0
    for ok in match:
        if ok:
            cnt += 1
        else:
            cnt = 0
        ans = max(ans, cnt)
print(ans)
