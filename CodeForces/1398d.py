"""
参考：https://twitter.com/laycrs/status/1294311698526961664
・自力ならず。。これは悔しい。
・ソートしてDP
・制約が「貪欲するな」って言ってたのにプリキューに突っ込んでみたり、
　何を思ったかフローを疑ったり、迷走しまくって終わった。。。
・確かにソートした時点で、ある種類の中でどれから使うのが最善かは決まっている訳で、
　あとは3種類からどの2種類を使うかの3通りしか遷移はないんだな。。
　なんか3次元DPちょっと考えたんだけど、遷移がもっとたくさんあってもう1重ループ回る気がして、
　じゃあ間に合わないな、ってそれ以上考えないで辞めちゃったんだよね。。
・方針分かったら結構実装はすぐ。最初MLEして、INFの値を10^9まで減らしたら無事AC。
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
INF = 10 ** 9
MOD = 10 ** 9 + 7

N, M, L = MAP()
R = LIST()
G = LIST()
B = LIST()

R.sort(reverse=1)
G.sort(reverse=1)
B.sort(reverse=1)
dp = list3d(N+1, M+1, L+1, -INF)
dp[0][0][0] = 0
for i in range(N+1):
    for j in range(M+1):
        for k in range(L+1):
            if i < N and j < M:
                dp[i+1][j+1][k] = max(dp[i+1][j+1][k], dp[i][j][k] + R[i]*G[j])
            if j < M and k < L:
                dp[i][j+1][k+1] = max(dp[i][j+1][k+1], dp[i][j][k] + G[j]*B[k])
            if i < N and k < L:
                dp[i+1][j][k+1] = max(dp[i+1][j][k+1], dp[i][j][k] + R[i]*B[k])
ans = 0
for i in range(N+1):
    for j in range(M+1):
        for k in range(L+1):
            ans = max(ans, dp[i][j][k])
print(ans)
