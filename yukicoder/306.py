"""
・自力AC
・三分探索(小数)、誤差
・普通にfloatでhypotしたら誤差死した。Decimalがsqrtを持っているのを知っていたので事なきを得た。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def calc(y):
    return (x1**2 + (y1-y)**2).sqrt() + (x2**2 + (y2-y)**2).sqrt()

x1, y1 = MAP()
x2, y2 = MAP()

if y2 < y1:
    y1, y2 = y2, y1
    x1, x2 = x2, x1
x1, y1 = Decimal(x1), Decimal(y1)
x2, y2 = Decimal(x2), Decimal(y2)

low = y1
hi = y2
for i in range(1000):
    m1 = (low*2+hi) / 3
    m2 = (low+hi*2) / 3
    res1 = calc(m1)
    res2 = calc(m2)
    if res1 <= res2:
        hi = m2
    else:
        low = m1
y = m1
print(y)
