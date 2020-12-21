"""
・手元でちょっと試してフィボナッチな動きをしたのでDPしてしまったけど、
　すぐ大きくなるんだから愚直シミュレーションすればよかったんだよね。。
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

for _ in range(INT()):
    a, b, N = MAP()

    if a > b:
        a, b = b, a
    dp = [0] * 1000
    dp[0] = a
    dp[1] = b
    i = 1
    while dp[i] <= N:
        dp[i+1] = dp[i] + dp[i-1]
        i += 1
    ans = i - 1
    print(ans)
