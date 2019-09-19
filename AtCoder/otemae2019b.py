# -*- coding: utf-8 -*-

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
INF = float('inf')
MOD = 10 ** 9 + 7

M, N, K = MAP()
A = LIST()

B = [0] * M
for a in A:
    B[a-1] += 1

# Mマスの各位置に駒を集めてみる
ans = 0
for i in range(M):
    cnt = B[i]
    for j in range(1, K+1):
        # 前後にKマスの範囲で、移動させてこれるか確認
        if (i-j >= 0 and B[i-j] != 0) \
                or (i+j < M and B[i+j] != 0):
            cnt += 1
    ans = max(ans, cnt)
print(ans)
