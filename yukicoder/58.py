"""
・自力AC
・確率系
・DPする。誤差制約が緩いので、想定解でモンテカルロ法ってのがあるみたい。でもまあDPでもいいよね。
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

N = INT()
K = INT()

# i個目のサイコロまで見て、太郎の得点-二郎の得点+60=jとした時のその状態になる通り数
dp = list2d(N+1, 120, 0)
dp[0][60] = 1
for i in range(N):
    for j in range(120):
        if dp[i][j] == 0:
            continue
        # 二郎の振るサイコロ
        for k in range(1, 7):
            if i < K:
                # 太郎の振るサイコロ(イカサマ)
                for l in [4, 4, 5, 5, 6, 6]:
                    dp[i+1][j+l-k] += dp[i][j]
            else:
                # 太郎の振るサイコロ(通常)
                for l in range(1, 7):
                    dp[i+1][j+l-k] += dp[i][j]
win = 0
for j in range(61, 120):
    # 最終的に勝ち状態の通り数を集める
    win += dp[N][j]
total = sum(dp[N])
ans = win / total
print(ans)
