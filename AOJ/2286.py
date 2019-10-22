# -*- coding: utf-8 -*-

"""
・蟻本演習2-6-4
・素数応用
・とりあえず自力やってみたけどこれはWA
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

N = 1000001
table = [0] * (N+1)
for i in range(2, N+1):
    # 各素数から、自分の倍数になる数に、自分が重複させる数を足し込んでいく
    if table[i] == 0:
        cnt = 1
        for j in range(i*2, N+1, i):
            table[j] += cnt
            cnt += 1

ans = [0] * (N+1)
ans[0] = 1
ans[1] = 2
for i in range(2, N+1):
    ans[i] = ans[i-1] + (i-1) - table[i]

for i in range(INT()):
    a = INT()
    print(ans[a])
