"""
・自力AC！
・数え上げDP
・想定解数学っぽいよなーと思いつつも、DPでぶん殴った。
・状態を3つ、まだ0も9も出てない、どっちか出た、両方出た、で管理する。
・あとはそれぞれの状態に従って、0~9の10通り遷移させていけばOK。
・想定解、包除だって。それで茶diffってみんな最近数学力高すぎ。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
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

dp = list2d(3, N+1, 0)
dp[0][0] = 1
for i in range(N):
    # 0,9以外
    dp[0][i+1] += dp[0][i] * 8
    # 0,9
    dp[1][i+1] += dp[0][i] * 2
    # 0,9以外と、0,9のうち既に出た方
    dp[1][i+1] += dp[1][i] * 9
    # 0,9のまだ出てない方
    dp[2][i+1] += dp[1][i]
    # 全部出てるので全部変化なし
    dp[2][i+1] += dp[2][i] * 10
    dp[0][i+1] %= MOD
    dp[1][i+1] %= MOD
    dp[2][i+1] %= MOD
ans = dp[2][N]
print(ans)
