"""
・自力AC！
・ANDが条件の行・列は絶対1でないといけないのでまず1埋めする。
　この状態で、OR条件から1にしないといけない行・列を確認して、
　少ない方は多い方と被せられるのでmax(行, 列)をさっきのANDの分に足す。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

N = INT()
A = LIST()
B = LIST()

rows = [0] * N
cols = [0] * N
ans = 0
for i in range(N):
    for j in range(N):
        if A[i] == 2 or B[j] == 2:
            ans += 1
            rows[i] = 1
            cols[j] = 1

cnt1 = 0
for i in range(N):
    if A[i] == 1 and rows[i] == 0:
        cnt1 += 1
cnt2 = 0
for j in range(N):
    if B[j] == 1 and cols[j] == 0:
        cnt2 += 1
ans += max(cnt1, cnt2)
print(ans)
