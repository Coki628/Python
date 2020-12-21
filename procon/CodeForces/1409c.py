"""
・自力AC
・全探索、約数列挙
・差分として使う値は約数取ったけど、制約小さいから、間の数字全部試してもいけるのかもな。
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
INF = 10**19
MOD = 10**9 + 7

def divisors(N: int) -> set:
    from math import sqrt

    s = {1, N}
    for i in range(2, int(sqrt(N))+1):
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

def check(d, x, y):
    arr = []
    i = y
    while len(arr) < N and i > 0:
        arr.append(i)
        i -= d
    i = y + d
    while len(arr) < N:
        arr.append(i)
        i += d
    if x not in arr:
        return [INF]
    arr.sort()
    return arr

for _ in range(INT()):
    N, x, y = MAP()

    dist = y - x
    divs = divisors(dist)
    mn = INF
    ans = []
    for div in divs:
        res = check(div, x, y)
        if res[-1] < mn:
            mn = res[-1]
            ans = res
    print(*ans)
