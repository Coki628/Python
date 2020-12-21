"""
・LCM、約数列挙
・題意を満たすのは、Nの約数で自分以外の一番大きいやつと、残ったやつのペアを作る。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def divisors(N: int) -> set:
    from math import sqrt

    s = {1, N}
    for i in range(2, int(sqrt(N))+1):
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

for _ in range(INT()):
    N = INT()

    divs = sorted(divisors(N), reverse=1)[1:]
    a = divs[0]
    b = N - divs[0]
    print(a, b)
