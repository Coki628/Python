"""
参考：https://img.atcoder.jp/abc169/editorial.pdf
　　　https://www.youtube.com/watch?v=-fTsuyin-a8&feature=youtu.be
・自力ならず。。
・部分和DP応用
・今回の、全体集合の部分集合Tの部分集合U、を考える時に各要素aが取りうる遷移として、
　1. TにもUにも含まれない
　2. Tには含まれるがUには含まれない
　3. TにもUにも含まれる
　の3通りがある。ここで1,2は答えとなる集合に変化がないので、いわゆる普通の部分和DPでいう
　「この要素を使わない」の遷移、3が「この要素を使う」の遷移になる。
・まとめると、要素を使わない方の遷移を2倍やって後は普通、だ。
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
MOD = 998244353

N, S = MAP()
A = LIST()

dp = list2d(N+1, S+1, 0)
dp[0][0] = 1
for i, a in enumerate(A):
    for j in range(S+1):
        dp[i+1][j] += dp[i][j] * 2
        dp[i+1][j] %= MOD
        if j+a <= S:
            dp[i+1][j+a] += dp[i][j]
            dp[i+1][j+a] %= MOD
ans = dp[N][S]
print(ans)
