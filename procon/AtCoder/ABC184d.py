"""
参考：ツイート
・自力ならず。。
・確率DP、期待値
・一応、確率DPって言われたら、ガチャガチャやった末、実装は自力で作れた。
・確率DPの基本方針として、初期状態を100% = 1 として、
　遷移先 += 遷移元 * その遷移をする確率、とやっていく。
・確率DPで、その状態になる確率みたいなやつを網羅すると、
　操作回数が何回で終了した確率も集められるので、その回数を掛けて期待値にできる。
　詳細はソース内コメント。
・手前にあるとはいえ、これのdiffがE,Fよりは低いんだよな。
　確率DPとか期待値DPは独特のクセがあるから、練習積まないとダメだなー。
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

a, b, c = MAP()

N = 100
M = N-a + N-b + N-c

# dp[i][j][k] := 袋に金貨i枚、銀貨j枚、銅貨k枚が入った状態になる確率
dp = list3d(N+1, N+1, N+1, 0)
# 初期状態を100%にする
dp[a][b][c] = 1
for i in range(a, N+1):
    for j in range(b, N+1):
        for k in range(c, N+1):
            # いずれかが100枚になっていたら終了なのでそれ以上遷移させない
            if i == N or j == N or k == N:
                continue
            # この状態でのコイン総枚数
            total = i + j + k
            if i+1 <= N:
                # 金貨を取る
                dp[i+1][j][k] += dp[i][j][k] * i / total
            if j+1 <= N:
                # 銀貨を取る
                dp[i][j+1][k] += dp[i][j][k] * j / total
            if k+1 <= N:
                # 銅貨を取る
                dp[i][j][k+1] += dp[i][j][k] * k / total

# C[i] := 操作回数i回になる確率
C = [0] * (M+1)
for i in range(a, N+1):
    for j in range(b, N+1):
        for k in range(c, N+1):
            if i-a >= 0 and j-b >= 0 and k-c >= 0:
                # 操作回数
                cnt = i-a + j-b + k-c
                # 金貨100枚で終了
                if i == N and j != N and k != N:
                    C[cnt] += dp[i][j][k]
                # 銀貨100枚で終了
                if i != N and j == N and k != N:
                    C[cnt] += dp[i][j][k]
                # 銅貨100枚で終了
                if i != N and j != N and k == N:
                    C[cnt] += dp[i][j][k]

ans = 0
for i in range(M+1):
    # 操作回数の期待値なので、操作回数*そうなる確率
    ans += i*C[i]
print(ans)
