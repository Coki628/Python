# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・上がってるか下がってるかフラグ切り替えて左端sを適切に制御する。
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

N = INT()
A = [INT() for i in range(N)]

is_up = False
s = u = 0
ans = 1
for i in range(N-1):
    if A[i] < A[i+1]:
        if not is_up:
            is_up = True
            s = i
    else:
        is_up = False    
        u = i + 1
        ans = max(ans, u-s+1)
else:
    ans = max(ans, N-s)
print(ans)
