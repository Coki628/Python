# -*- coding: utf-8 -*-

"""
・自力AC
・条件判定ミスって何度かWAした。。
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

def check(x):
    s = str(abs(x))
    n = len(s)
    return n >= 3 and s.count('0') == n-1

a, b = MAP()

if check(a) and check(b):
    ans = a * b // 10
    print(ans)
else:
    ans = a * b
    if abs(ans) >= 100000000:
        print('E')
    else:
        print(ans)
