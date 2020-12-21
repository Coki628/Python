# -*- coding: utf-8 -*-

"""
参考：https://qiita.com/drken/items/b97ff231e43bce50199a
・蟻本演習2-6-2、AOJ本
・拡張ユークリッドの互除法
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
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

# 拡張ユークリッドの互除法(ax+by=gcd(a, b)の解を求める)
def extgcd(a, b, x=1, y=0):
    if b == 0:
        x = 1
        y = 0
        return (y, x)
    else:
        x, y = extgcd(b, a%b, y, x)
        y -= a // b * x
        return (y, x)

a, b = MAP()

y, x = extgcd(a, b)
print(x, y)
