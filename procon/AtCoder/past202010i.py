"""
・累積和にぶたん
・円環なので2倍作って開始位置全部試す。
・題意を満たすには、全体の約半分に近いものが取れればいいけど、
　細かいとこ詰めるの大変だから、探索位置は半分切り捨てと切り上げ両方、
　確認するのはその位置とひとつ手前で計4箇所確かめながらやった。
"""

import sys
from itertools import accumulate
from bisect import bisect_right

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

total = sum(A)
A += A
acc = [0] + list(accumulate(A))
ans = INF
for i in range(N):
    j = bisect_right(acc, acc[i]+total//2)
    half = acc[j] - acc[i]
    ans = min(ans, abs((total-half) - half))
    half = acc[j-1] - acc[i]
    ans = min(ans, abs((total-half) - half))
    j = bisect_right(A, acc[i]+ceil(total, 2))
    half = acc[j] - acc[i]
    ans = min(ans, abs((total-half) - half))
    half = acc[j-1] - acc[i]
    ans = min(ans, abs((total-half) - half))
print(ans)
