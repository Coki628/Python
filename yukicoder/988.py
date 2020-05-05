# -*- coding: utf-8 -*-

"""
・自力AC
・ちょっと式変形
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

N, M, MOD = MAP()
op, *li = input().split()
B = list(map(int, li))
A = LIST(N)

ans = 0
if op == '+':
    asm = sum(A) % MOD
    bsm = sum(B) % MOD
    ans = (asm * M + bsm * N) % MOD
else:
    bsm = sum(B) % MOD
    for a in A:
        # 各行について、a1*(b1+b2+b3+...) みたいになる
        ans += a * bsm
        ans %= MOD
print(ans)
