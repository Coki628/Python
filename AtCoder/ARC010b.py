# -*- coding: utf-8 -*-

"""
・自力AC、思いの外、実装時間かかった。
・日付管理。実務で使いそうな感じ。
"""

import sys
from datetime import date, timedelta

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

holidays = {}
for i in range(1, 13):
    holidays[i] = set()

for i in range(N):
    m, d = map(int, input().split('/'))
    y = 2012
    day = date(y, m, d)
    while y == 2012:
        wd = day.weekday()
        # 土曜でも日曜でも祝日でもない
        if wd != 5 and wd != 6 and d not in holidays[m]:
            holidays[m].add(d)
            break
        # 翌日にする
        day = day + timedelta(days=1)
        y, m, d = day.year, day.month, day.day

# 1年間を通して、一番連続するところを見つける
y, m, d = 2012, 1, 1
day = date(y, m, d)
mx = cnt = 0
for i in range(366):
    wd = day.weekday()
    if wd == 5 or wd == 6 or d in holidays[m]:
        cnt += 1
    else:
        mx = max(mx, cnt)
        cnt = 0
    day = day + timedelta(days=1)
    y, m, d = day.year, day.month, day.day
mx = max(mx, cnt)

print(mx)
