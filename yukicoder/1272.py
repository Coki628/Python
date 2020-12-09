"""
・級数
・こっちも足しまくる方針でAC。。
・分母が冪乗ですぐでかくなるので、すぐにほぼ無みたいになる。
　適当に10まで足したらサンプル合ってたので投げたら無事AC。。
"""

import sys
from math import sin

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
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

k = INT()

ans = 0
for i in range(1, 10):
    ans += sin(k*i) / (i**i)
print(ans)
