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

N=INT()
A=LIST()

cnt=1
for i in range(N):
    if A[i]%2==0:
        # 自身が偶数なら似ている奇数は2つ
        cnt*=2
    else:
        # 自身が奇数なら1つ
        cnt*=1

# 全体 - 全ての項の積が奇数
print(3**N-cnt)
