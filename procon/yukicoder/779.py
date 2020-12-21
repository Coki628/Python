# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・こういう現実の日付使う系はpythonの標準ライブラリが優秀で感謝。
"""

import sys
from datetime import datetime

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

y, m, d = MAP()

ymd = datetime(y, m, d)
l = datetime(1989, 1, 8)
r = datetime(2019, 4, 30)

if l <= ymd <= r:
    Yes()
else:
    No()
