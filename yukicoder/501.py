"""
・さくっと自力AC
・辞書順最小、場合分け
・とにかくAを最優先。辞書順最小だから。
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
INF = 10 ** 18
MOD = 10 ** 12
EPS = 10 ** -10

N, D = MAP()

diff = N - D
if diff == 0:
    ans = 'A' * N
elif diff < 0:
    ans = 'A' * (N-abs(diff)) + 'B' * abs(diff)
else:
    ans = 'A' * (N-diff) + 'C' * diff
print(ans)
