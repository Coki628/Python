# -*- coding: utf-8 -*-

"""
参考：http://xxxasdfghjk999.hatenablog.jp/entry/2018/07/26/180904
　　　https://mathtrain.jp/phi
・蟻本演習2-6-4
・素数応用
・ファレイ数列、オイラーのファイ関数
"""

import sys
from itertools import accumulate

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
table = list(range(N+1))
for i in range(2, N+1):
    # 各素数から、自分の倍数になる数に、オイラーのファイ関数を掛け込んでいく
    if table[i] == i:
        for j in range(i, N+1, i):
            table[j] *= 1 - 1/i

# 0/0の分を入れる
table[0] = 1
ans = list(accumulate(table))
for i in range(INT()):
    a = INT()
    print(int(ans[a]))
