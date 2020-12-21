"""
・ちょっと遷移の条件ゴチャゴチャしてるけど、普通の1次元DP。
・特にゴールの瞬間は他と扱いが変わるので丁寧に場合分けする。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

N, L = MAP()
obs = [0] * (L+1)
A = LIST()
T1, T2, T3 = MAP()
for a in A:
    obs[a] = T3

dp = [INF] * (L+1)
dp[0] = 0
for i in range(L):
    # 走る
    dp[i+1] = min(dp[i+1], dp[i]+T1+obs[i+1])
    # 小ジャンプ
    if i+2 <= L:
        dp[i+2] = min(dp[i+2], dp[i]+T1+T2+obs[i+2])
    else:
        dp[L] = min(dp[L], dp[i]+T1//2+T2//2)
    # 大ジャンプ
    if i+4 <= L:
        dp[i+4] = min(dp[i+4], dp[i]+T1+T2*3+obs[i+4])
    elif i+3 == L:
        dp[L] = min(dp[L], dp[i]+T1//2+T2*5//2)
    elif i+2 == L:
        dp[L] = min(dp[L], dp[i]+T1//2+T2*3//2)
    else:
        dp[L] = min(dp[L], dp[i]+T1//2+T2//2)
ans = dp[L]
print(ans)
