"""
参考：みんなのツイート
・自力ならず。。こういうこどふぉギャグほんとつらい。
・さっさと実験すればよかった。
・やってみると、答えは常に一定で、数列の総和になる。。
"""

import sys
from itertools import permutations

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
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
EPS = 10**-10

for _ in range(INT()):
    N, K = MAP()
    A = LIST()

    # for perm in permutations(A):
    #     ans = 0
    #     for i in range(N):
    #         for j in range(i, N):
    #             ans += perm[j] / (j+1)
    #     print(ans)

    if sum(A) == K:
        YES()
    else:
        NO()
