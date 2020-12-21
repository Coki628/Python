# -*- coding: utf-8 -*-

"""
・これは愚直にやってるTLE想定。
・不等号の兼ね合いでWA取るのに手間取った。
・ここから高速化する。
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

T1, T2 = MAP()
A1, A2 = MAP()
B1, B2 = MAP()

def calc(a, b):
    cnt = 0
    a1 = a + T1 * A1
    b1 = b + T1 * B1
    if a > b and a1 <= b1 or a < b and a1 >= b1:
        cnt += 1
    a2 = a1 + T2 * A2
    b2 = b1 + T2 * B2
    if a1 > b1 and a2 <= b2 or a1 < b1 and a2 >= b2:
        cnt += 1
    if a == b and a2 == b2:
        print('infinity')
        exit()
    return a2, b2, cnt

a = b = 0
ans = 0
while True:
    a, b, cnt = calc(a, b)
    if cnt == 0:
        break
    ans += cnt
print(ans)
