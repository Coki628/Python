# -*- coding: utf-8 -*-

"""
・等差数列の和
・pythonTLE,pypyAC1.3秒。
・100万の前計算の中でMOD除算のlogが入ってるから重いんだと思う。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

# modの除算(フェルマーの小定理)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

def get_sum(a, b, c):
    """ 等差数列の和：(初項a, 末項b, 項数c) """
    return fermat((a+b) * c, 2, MOD)

a = INT()
b = INT()

N = 10**6
A = [0] * (N+1)

for i in range(1, N+1):
    A[i] = get_sum(i, i**2, i)

ans = 0
for i in range(a, b+1):
    ans += A[i]
print(ans%MOD)
