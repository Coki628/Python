"""
・さくっと自力AC
・桁和
・書いてある通りにやってOK。解説にもっと高速な解法の話がある。
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

def digit_sum(n):
    """ 桁和：O(logN) """

    ans = 0
    while n > 0:
        ans += n % 10
        n //= 10
    return ans

N = INT()

M = 100
dp = [0] * M
dp[0] = N
for i in range(M-1):
    dp[i+1] = digit_sum(dp[i])
ans = dp[-1]
print(ans)
