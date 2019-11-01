# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-3
・反転(ライツアウト)
・WA。一番長い区間を起点にして広げていく方針を取ってみたけど違った。。
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

S = list(input())
N = len(S)

# ランレングス圧縮
A = []
cnt = 1
for i in range(N-1):
    if S[i] == S[i+1]:
        cnt += 1
    else:
        A.append(cnt)
        cnt = 1
A.append(cnt)

# 例外処理
if len(A) <= 2:
    print(max(A))
    exit()

M = len(A)
ans = max(A[0], A[-1])
for i in range(1, M-1):
    ans = max(ans, min(A[i]+A[i-1], A[i]+A[i+1]))
print(ans)
