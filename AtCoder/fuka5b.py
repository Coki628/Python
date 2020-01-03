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

while 1:
    instr = input()
    if instr == '0':
        break

    ymd, hms = instr.split()
    y, mo, d = map(int, ymd.split('/'))
    h, mi, s = map(int, hms.split(':'))
    add = int(input(), 2)

    # 大き過ぎる年を分解
    di, y = divmod(y, 4000)
    # 年=0がdatetime範囲外なので調整
    if y == 0:
        y += 4000
        di -= 1

    # 必要な日時の作成
    t = datetime(y, mo, d, h, mi, s)
    t = str(t + timedelta(seconds=add))

    # 分解した年を戻す
    y = int(t[:4])
    y += di * 4000
    t = str(y) + t[4:]

    # 整形して出力
    t = t.replace('-', '/')
    print(t)
