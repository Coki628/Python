# -*- coding: utf-8 -*-

"""
・逆順ソートして大きい方から貪欲
・でもこれ嘘解法かもしれん。
　公式解が全探索かDPだった。。
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

N = INT()
A = [INT() for i in range(N)]

cnt1 = 0
cnt2 = 0
A.sort(reverse=True)
for i in range(N):
    if cnt1 <= cnt2:
        cnt1 += A[i]
    else:
        cnt2 += A[i]
print(max(cnt1, cnt2))
