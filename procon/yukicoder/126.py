"""
・自力ならず。。悔しい。
・場合分け
・考えたけど詰めきれず。。条件あと1つ足りないのが出なかった。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

a, b, x = MAP()

if x == 1 or abs(x-a) <= abs(x-b):
    ans = abs(x-a) + x
else:
    # Aが地下1階にいる時はBがそこに行けないので1階に行くしかない
    if a == 0:
        ans = abs(x-b) + abs(x-1) + abs(a-1) + 1
    else:
        ans = min(
            abs(x-b) + abs(x-a) + a,
            abs(x-b) + abs(x-1) + abs(a-1) + 1,
        )
print(ans)
