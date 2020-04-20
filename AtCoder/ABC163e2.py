# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/prd_xxx/status/1251872733803118593
　　　https://atcoder.jp/contests/abc163/submissions/12137623
　　　https://img.atcoder.jp/abc163/editorial.pdf
・自力ならず。
・2乗の区間DP
・3乗のやつは何度かやったことあるけど、2乗のやつは多分初めてやった。
・今何人目を見ているか、が分かっていれば、左右合わせて何人確定してるか、は一意なので、
　左に0,右にk 〜 左にk,右に0 までを見る感じで、内側が1重ループで済む。
・ほーんと、DPは深いなぁ。やってない感じのやつがまだたくさんあるな。
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

N = INT()
A = LIST()

A = [(a, i) for i, a in enumerate(A)]
A.sort(reverse=1)

# dp[i][j] := 左にi人、右にj人確定した状態での最大値
dp = list2d(N+1, N+1, -INF)
dp[0][0] = 0
for i, (a, x) in enumerate(A):
    # j := 今左に確定してる人数
    for j in range(i+1):
        # k := 今右に確定してる人数
        k = i - j
        # このaを左にする遷移
        y = j
        dp[j+1][k] = max(dp[j+1][k], dp[j][k] + abs(x - y) * a)
        # このaを右にする遷移
        y = N - 1 - k
        dp[j][k+1] = max(dp[j][k+1], dp[j][k] + abs(x - y) * a)
ans = 0
for i in range(N+1):
    # 左右に計N人決まった状態での最大を取る
    ans = max(ans, dp[i][N-i])
print(ans)
