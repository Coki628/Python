"""
・そもそも10万数列の作成1回にしたらこれもAC。でも1.4秒。
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
INF = 10 ** 10
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

nums = sorted(set(A), reverse=1)
ans = -INF
acc = [0] * (N+1)
accmn = [0] * (N+1)
for num in nums:
    for i in range(N):
        if A[i] > num:
            A[i] = -INF
    for i in range(N):
        acc[i+1] = acc[i] + A[i]
    for i in range(N):
        accmn[i+1] = min(accmn[i], acc[i])
    mx = -INF
    for i in range(N):
        mx = max(mx, acc[i+1]-accmn[i])
    ans = max(ans, mx-num)
print(ans)
