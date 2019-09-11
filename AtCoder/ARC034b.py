# -*- coding: utf-8 -*-

"""
・調査用
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

def f(x):
    x = str(x)
    res = 0
    for s in x:
        res += int(s)
    return res

for i in range(1, 301):
    res = i + f(i)
    print(i, res)

for i in [999,1000,1999,2000,2999,3000,9999,10000,19999,20000]:
    res = i + f(i)
    print(i, res)

# 各桁が1増える毎に変化する値は決まっている
# 1=>2,10=>11,10=>101,100=>1001..以後、0が増えていく
