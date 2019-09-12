# -*- coding: utf-8 -*-

"""
・さくっと自力AC
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

N = INT()
A = [INT() for i in range(N)]

# 全部同じ色なら終わらない
if A.count(1) == N or A.count(0) == N:
    print(-1)
    exit()


# 環状なので2つ繋げて考える
A = A + A
mx = cnt = 1
for i in range(1, N*2):
    if A[i-1] == A[i]:
        cnt += 1
    else:
        # 一番長い区間が一番時間がかかる
        mx = max(mx, cnt)
        cnt = 1
mx = max(mx, cnt)

print(ceil(mx, 2))
