"""
・ちょっとDP、前計算、二分探索
・でも前計算したDP配列が26000くらいでTが1000だから普通にループ回しても間に合った。。
"""

import sys
from bisect import bisect_right

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

MAX = 10 ** 9
dp = [0]
a = 2
b = 0
while dp[-1]+a+b <= MAX:
    dp.append(dp[-1]+a+b)
    a += 2
    b += 1
M = len(dp)

for _ in range(INT()):
    N = INT()

    ans = 0
    # for i in range(M-1, 0, -1):
    #     if dp[i] <= N:
    #         N -= dp[i]
    #         ans += 1
    while 1:
        idx = bisect_right(dp, N) - 1
        if idx == 0:
            break
        N -= dp[idx]
        ans += 1
    print(ans)
