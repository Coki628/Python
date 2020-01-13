# -*- coding: utf-8 -*-

"""
・終了後ほぼ自力AC
・全操作の総和を数え上げる系。これ3日間のコンテストで全部出た。。(ABC150e,dwacon2020qual_b,ABC151e)
・コンテスト中はもやっとしたまま終わってしまったけど、今実験しながらやったらすぐ見えた。
　さっさと実験すれば良かったなぁ。。
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

def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

N = INT()
A = LIST()

# 大きい数ほど少なく使うのが最適なので降順ソートしておく
A.sort(reverse=1)
# 各要素が変更のために何回使われるか数える
cnts = [0] * N
cnts[0] = pow(4, N-1, MOD)
add = fermat(cnts[0], 2, MOD)
for i in range(1, N):
    cnts[i] = cnts[i-1] + add
    cnts[i] %= MOD

ans = 0
for i in range(N):
    ans += A[i] * cnts[i]
    ans %= MOD
print(ans * 2 % MOD)
