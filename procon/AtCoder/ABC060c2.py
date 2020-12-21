# -*- coding: utf-8 -*-

"""
・今日の一問から
・後ろからやったけど、これは前からでもよかったね。
　まあ速攻で決めれたからOK。
"""

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

N, T = MAP()
A = LIST()

ans = T
for i in range(N-1, 0, -1):
    ans += min(T, A[i]-A[i-1])
print(ans)
