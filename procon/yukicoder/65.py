"""
・自力AC
・期待値DP
・操作が終了する所を0として、前の値+1しながら開始位置に戻っていく。期待値DPでよく見る方針。(past201912oとか)
・今回は練習問題って感じで遷移も分かりやすかったから解けたけど、難しくなった時ちゃんと対応していきたい。
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
EPS = 10 ** -10

K = INT()

# これまでの目の合計がiのとき、合計がKに達するまでにこの先振ることになる回数の期待値
dp = [INF] * (K+6)
# 合計がK以上になったら終了なので0
for i in range(K, K+6):
    dp[i] = 0
# K-1以前に遡っていく
for i in range(K-1, -1, -1):
    dp[i] = 1
    for j in range(i+1, i+7):
        dp[i] += dp[j] / 6
ans = dp[0]
print(ans)
