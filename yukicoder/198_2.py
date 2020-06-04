"""
参考：http://sugarknri.hatenablog.com/entry/2016/05/18/152712
・中央値から求める解。
・手持ち数から、達成できる値の上限を(sum(A)+b)//Nとする。これと中央値でmin取ればOK。
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

b = INT()
N = INT()
A = LIST(N)

limit = (sum(A)+b) // N
A.sort()
if N % 2 == 1:
    mid = A[N//2]
    mid = min(mid, limit)
    ans = 0
    for i in range(N):
        ans += abs(mid - A[i])
    print(ans)
else:
    mid1 = A[N//2-1]
    mid2 = A[N//2]
    mid1 = min(mid1, limit)
    mid2 = min(mid2, limit)
    ans1 = ans2 = 0
    for i in range(N):
        ans1 += abs(mid1 - A[i])
        ans2 += abs(mid2 - A[i])
    print(min(ans1, ans2))
