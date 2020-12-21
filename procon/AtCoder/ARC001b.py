# -*- coding: utf-8 -*-

"""
・今日の一問から
・300点相当自力AC
・がしかし条件ゴチャゴチャしててスッキリいかなかった。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

A, B = MAP()

dif = abs(A-B)

# 10未満まで近づける
cnt = dif // 10
dif %= 10

# 5未満まで近づける
# 5のが早い
if dif <= 7:
    cnt += dif // 5
    dif %= 5
# 10のが早い
else:
    cnt += 1
    dif = 10 - dif

# 5のが早い
if dif == 4:
    cnt += 2
# 1のが早い
else:
    cnt += dif

print(cnt)
