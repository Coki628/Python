# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/nagitausu/status/1221428354013773824
・500点自力ならず。ド典型でこれは悔しい。。
・個数制限なしナップザック
・制約のNが1万あるから2次元DPきついよなぁってぱっと考えてしまったんだけど、
　N1万でHが1000だからN*Hでも1000万で大丈夫じゃんね。。
・あと、同じ魔法何回でもだと、毎回遷移がN回になって無理とかも思ったんだけど、
　これも順序関係ないから状態まとまるやつじゃんね。。
・やったことある典型は気付けるようになりたいよなぁ。。。
・計算量N*max(A)*2くらいで2000万くらい。pypyAC0.9秒。
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

H, N = MAP()
AB = [()]
for i in range(N):
    a, b = MAP()
    AB.append((a, b))

# dp[i][j] := i個目の魔法まで見て、ダメージjを与える時の最小消費魔力
dp = list2d(N+1, 20007, INF)
dp[0][0] = 0
for i in range(1, N+1):
    a, b = AB[i]
    # Hぴったりとは限らないので、十分先まで遷移させる
    for j in range(20007):
        # この魔法を使わない
        dp[i][j] = min(dp[i][j], dp[i-1][j])
        if j-a >= 0:
            # この魔法を使う(再利用・初回)
            dp[i][j] = min(dp[i][j], dp[i][j-a] + b, dp[i-1][j-a] + b)
print(min(dp[N][H:]))
