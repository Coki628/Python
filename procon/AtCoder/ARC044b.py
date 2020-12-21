# -*- coding: utf-8 -*-

"""
・自力WA。。
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
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

if A[0] != 0:
    print(0)
    exit()
for i in range(1, N):
    if A[i] == 0:
        print(0)
        exit()

M = max(A)
C = Counter(A)
ans = 1
for i in range(1, M+1):
    if C[i] == 0:
        print(0)
        exit()
    # 同じ距離同士の辺の張り方の通り数 * ひとつ前の距離と繋がる辺の張り方の通り数
    ans *= pow(2, C[i]-1, MOD) * pow(pow(2, C[i-1], MOD)-1, C[i], MOD)
    ans %= MOD
print(ans)
