# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・速攻自力AC
・ビット全探索
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

S = list(input())
N = len(S) - 1

# 間に+を挟むかどうかでビット全探索
ans = 0
for bit in range(1<<N):
    s = ''
    for i, num in enumerate(S):
        if bit & 1<<i:
            s += num + '+'
        else:
            s += num
    ans += eval(s)
print(ans)
