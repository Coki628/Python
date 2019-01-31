# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc054/editorial.pdf
・3次元DP
・配るDP
・添字の当て方までは合ってたのに、中の処理がうまく作れなかった…。
・pythonで1.8秒くらい、pypyにしたら0.5秒くらいですごく速くなった。
・計算量はO(40*400*400+400*400=656万くらい)
"""
 
import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7
 
N, Ma, Mb = map(int, input().split())
l = [[0] * 3 for i in range(N)]
for i in range(N):
    l[i][0], l[i][1], l[i][2] = map(int, input().split())

# 不可能な組み合わせは0とは区別するのでINF初期化
dp = [[[INF] * 401  for i in range(401)] for i in range(N+1)]
dp[0][0][0] = 0
for i in range(N):
    for j in range(401):
        for k in range(401):
            # あり得ない組み合わせの時は送らない
            if dp[i][j][k] == INF: continue
            dp[i+1][j][k] = min(dp[i][j][k], dp[i+1][j][k])
            dp[i+1][j+l[i][0]][k+l[i][1]] = min(dp[i][j][k] + l[i][2], dp[i+1][j+l[i][0]][k+l[i][1]])

ans = INF
# 条件に合うかの確認、これはdp[N]についてだけ回す
for j in range(1, 401):
    for k in range(1, 401):
        # j/k == Ma/Mb だと分母0が心配なので
        if j * Mb == k * Ma:
            ans = min(dp[N][j][k], ans)

if ans == INF:
    print(-1)
else:
    print(ans)
