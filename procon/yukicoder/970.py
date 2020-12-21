"""
・自力AC
・式変形
・全員が自分以外の全体から平均を取っているので、均すと全体の平均と同じになると思う。
　なので、元の数列と総和は等しいはず。
　すると、*(N-1)で自分以外の総和は分かるので、総和 - 自分*(N-1)
　をそれぞれについてやればいい。
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
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N = INT()
A = LIST()

sm = sum(A)
ans = [0] * N
for i, a in enumerate(A):
    ans[i] = sm - A[i] * (N-1)
print(*ans)
