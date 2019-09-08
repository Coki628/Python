# -*- coding: utf-8 -*-

"""
・終了後にちょっと整理した版
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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

N, K = MAP()
S = input()

# 初期状態でOKなものを数えておく
is_ok = [0] * N
for i in range(N):
    if S[i] == 'L':
        if i != 0 and S[i-1] == 'L':
            is_ok[i] = 1
    else:
        if i != N-1 and S[i+1] == 'R':
            is_ok[i] = 1

# LかRが連続する区間毎にまとめる
A = []
st = 0
for i in range(1, N):
    if S[i] != S[i-1]:
        A.append((st, i, S[i-1]))
        st = i
A.append((st, N, S[-1]))

# その区間を逆にしたらいくつ増やせるかを格納するLR
LR = {'L': [], 'R': []}
# ずっとLかRだけなら何も増やせない
if len(A) >= 2:
    for st, end, d in A:
        # 先頭と末尾だけは+1
        if st == 0 or end == N:
            LR[d].append(1)
        # 他の場所は+2できる
        else:
            LR[d].append(2)

LR['L'].sort(reverse=True)
LR['R'].sort(reverse=True)
# 大きい値から貪欲にK回変えてみて、Lを変えるとRを変えるの良い方
ans = sum(is_ok) + max(sum(LR['L'][:K]), sum(LR['R'][:K]))
print(ans)
