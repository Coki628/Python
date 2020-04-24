# -*- coding: utf-8 -*-

import sys
from datetime import datetime, timedelta

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

y, m, d = map(int, input().split('/'))

day1 = datetime(y, m, d)
day3 = day1 + timedelta(days=2)

ans = [0] * 3
ans[0] = str(day3.year).zfill(4)
ans[1] = str(day3.month).zfill(2)
ans[2] = str(day3.day).zfill(2)
print('/'.join(ans))
