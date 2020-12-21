"""
・自力AC
・累積和、2値化
・スムーズに考察できた。
・単調増加と単調減少は累積和を個別に作っておいてそれぞれで判定する。
・各位置が条件を満たすかでTrue/Falseしておいて、区間全体がTrueかどうかは
　区間和 == 区間の長さ かどうかで判定できる。
"""

import sys
from itertools import accumulate

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

inc = [0] * (N-1)
dec = [0] * (N-1)
for i in range(N-1):
    if A[i] <= A[i+1]:
        inc[i] = 1
    if A[i] >= A[i+1]:
        dec[i] = 1
inc = [0] + list(accumulate(inc))
dec = [0] + list(accumulate(dec))

Q = INT()
for i in range(Q):
    l, r = MAP()

    ln = r - l
    if ln == 0:
        print(1, 1)
        continue
    ans = [0] * 2
    if inc[r]-inc[l] == ln:
        ans[0] = 1
    if dec[r]-dec[l] == ln:
        ans[1] = 1
    print(*ans)
