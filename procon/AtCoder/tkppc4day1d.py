# -*- coding: utf-8 -*-

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

N = INT()
A = LIST()

if N == 1:
    print(0)
    exit()

# 0: 未確定, 1:上昇, 2:下降
dir = 0
cnt = 1
if A[0] < A[1]:
    dir = 1
    cnt += 1
elif A[0] > A[1]:
    dir = 2
    cnt += 1

for i in range(2, N):
    if A[i-1] < A[i] and dir != 1:
        dir = 1
        cnt += 1
    elif A[i-1] > A[i] and dir != 2:
        dir = 2
        cnt += 1

if dir == 0:
    print(0)
else:
    print(cnt)
