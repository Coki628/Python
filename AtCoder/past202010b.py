"""
・予定外にハマった。
・小数点以下の任意な切り捨てが、案外ちゃんとやらないとできない。普通にfloor使うと整数になっちゃうし。
"""

import sys
from decimal import Decimal
from math import floor

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

x, y = map(Decimal, input().split())

if y == 0:
    print('ERROR')
else:
    ans = x / y
    # 小数点以下2桁で切り捨て
    ans = floor(ans*10**2) / (10**2)
    # 表示を小数点以下2桁に整形
    ans = ans = '{:.2f}'.format(ans)
    print(ans)
