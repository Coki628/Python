# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/tpyneriver/status/1238824070570897409
・自力ならず。。
・別解
・精度が心配なら、pythonはDecimalを使うという方法がある。これでもACした。
"""

import sys
from decimal import Decimal

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

a, b, c = MAP()

if Decimal(a).sqrt() + Decimal(b).sqrt() < Decimal(c).sqrt():
    Yes()
else:
    No()
