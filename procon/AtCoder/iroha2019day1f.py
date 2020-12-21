# -*- coding: utf-8 -*-

import sys
from math import sqrt
from functools import reduce
from operator import mul

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

# 素因数分解(素数の持ち方変形版)
def factorize(num: int) -> list:
    res = []
    # 終点はルート切り捨て+1
    end = int(sqrt(num)) + 1
    for i in range(2, end):
        cnt = 0
        # 素因数分解：小さい方から割れるだけ割って素数をカウント
        while num % i == 0:
            num //= i
            res.append(i)
        # 1まで分解したら終了
        if num == 1:
            break
    # 最後に残ったnumは素数(ただし1^1は1^0なので数に入れない)
    if num != 1:
        res.append(num)
    return res

def prod(nums): return reduce(mul, nums, 1)

N,K=MAP()

A=sorted(factorize(N))
if len(A)>=K:
    B=A[:K-1]+[prod(A[K-1:])]
    print(*B)
else:
    print(-1)
