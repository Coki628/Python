"""
・defaultdictのところをビット管理で普通に8までの値として数列に持たせた。
　これでもTLEだった。。。C++でAC確認。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def popcount(x):
    x -= (x >> 1) & 0x55
    x = (x & 0x33) + ((x >> 2) & 0x33)
    x = (x + (x >> 4)) & 0xf
    return x & 0xf

H, W = MAP()
grid = [[]] * H
for i in range(H):
    grid[i] = list(input())

if H >= 4 and W >= 4:
    print(-1)
    exit()
if H == 1 or W == 1:
    print(0)
    exit()

if W > H:
    grid = list(zip(*grid))
    H, W = W, H

cur = int(''.join(grid[0]), 2)
if W == 2:
    dp = list2d(H, 4, INF)
    for nxt in range(4):
        dp[0][nxt] = popcount(cur ^ nxt)
    for i in range(1, H):
        cur = int(''.join(grid[i]), 2)
        for k in range(4):
            if k in (1, 2):
                dp[i][0] = min(dp[i][0], dp[i-1][k] + popcount(cur ^ 0))
                dp[i][3] = min(dp[i][3], dp[i-1][k] + popcount(cur ^ 3))
            else:
                dp[i][2] = min(dp[i][2], dp[i-1][k] + popcount(cur ^ 2))
                dp[i][1] = min(dp[i][1], dp[i-1][k] + popcount(cur ^ 1))
else:
    dp = list2d(H, 8, INF)
    for nxt in range(8):
        dp[0][nxt] = popcount(cur ^ nxt)
    for i in range(1, H):
        cur = int(''.join(grid[i]), 2)
        # to 000
        dp[i][0] = min(dp[i][0], dp[i-1][5] + popcount(cur ^ 0))
        dp[i][0] = min(dp[i][0], dp[i-1][2] + popcount(cur ^ 0))
        # to 001
        dp[i][1] = min(dp[i][1], dp[i-1][3] + popcount(cur ^ 1))
        dp[i][1] = min(dp[i][1], dp[i-1][4] + popcount(cur ^ 1))
        # to 010
        dp[i][2] = min(dp[i][2], dp[i-1][7] + popcount(cur ^ 2))
        dp[i][2] = min(dp[i][2], dp[i-1][0] + popcount(cur ^ 2))
        # to 011
        dp[i][3] = min(dp[i][3], dp[i-1][6] + popcount(cur ^ 3))
        dp[i][3] = min(dp[i][3], dp[i-1][1] + popcount(cur ^ 3))
        # to 100
        dp[i][4] = min(dp[i][4], dp[i-1][1] + popcount(cur ^ 4))
        dp[i][4] = min(dp[i][4], dp[i-1][6] + popcount(cur ^ 4))
        # to 101
        dp[i][5] = min(dp[i][5], dp[i-1][7] + popcount(cur ^ 5))
        dp[i][5] = min(dp[i][5], dp[i-1][0] + popcount(cur ^ 5))
        # to 110
        dp[i][6] = min(dp[i][6], dp[i-1][4] + popcount(cur ^ 6))
        dp[i][6] = min(dp[i][6], dp[i-1][3] + popcount(cur ^ 6))
        # to 111
        dp[i][7] = min(dp[i][7], dp[i-1][5] + popcount(cur ^ 7))
        dp[i][7] = min(dp[i][7], dp[i-1][2] + popcount(cur ^ 7))
ans = min(dp[-1])
print(ans)
