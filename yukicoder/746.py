"""
・自力AC
・この手の桁DP系で通り数じゃなくて小数になるの珍しいなー、って思ったら違った。
・総WAで誤差の条件書いてないし、何かおかしいと思ったらサンプルが既にめっちゃ周期作ってた。ギャグ。。
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

# mod = 7
# dp = list2d(N+1, mod, 0)
# dp[0][0] = 1
# for i in range(N):
#     for j in range(mod):
#         for k in range(10):
#             nxt = (j*10+k) % mod
#             dp[i+1][nxt] += dp[i][j]
#     for j in range(mod):
#         dp[i+1][j] /= 10
# ans = dp[N][0] - 10**-N
# print(ans)

if N == 0:
    print(0)
else:
    ans = '0.'
    s = '142857'
    while len(ans) <= N+2:
        ans += s
    ans = ans[:N+2]
    print(ans)
