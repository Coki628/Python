"""
・6で割れるだけ割って、できなくなったら2を掛けながらまた6で割れるだけ割る。
　3が足りなくなったり、余計な素数が残ったりするならNG。
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

for _ in range(INT()):
    N = INT()

    ans = 0
    while N % 6 == 0:
        N //= 6
        ans += 1
    while N % 2 == 1 and N % 3 == 0:
        N *= 2
        N //= 6
        ans += 2
    if N == 1:
        print(ans)
    else:
        print(-1)
