# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/70720
・グリッド、フィボナッチ
・制約が大きいのでHとWは独立で考える。
・とりあえず1次元で見ると、同じ色が最大1つだけ隣り合う通り数はフィボナッチと同じDPになる。
・2次元になる時、H*Wくらいになりそうな感じがするけど、実際に塗り分けを作ってみると、
　1回縦2つ同じ色が並んだら、その横はずっと同じように縦2つにしないといけないし、
　下の方の別の行で急に横2つにしたりもできないことが分かる。(やると3つ以上隣り合ってしまう)
・つまり1次元で数えたのと同じ通り数より多くはならない。
・ので、縦の通り数と横の通り数を足すだけ。
・全部1色ずつの格子状は両方で重複して数えてしまっているので-1、白黒反転させた分で*2。
・フィボナッチっぽい、までは気づいたんだけど、2次元になると膨大で複雑な通り数になる気がして、先に進めなかった。
　もっといくつか試しに塗り分けて、意外と単純でパターンが少ないことに気づけたら良かった。。
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
INF = float('inf')
MOD = 10 ** 9 + 7

H, W = MAP()

N = max(H, W)
dp = [0] * (N+2)
dp[1] = 1
dp[2] = 2
for i in range(3, N+1):
    dp[i] = dp[i-1] + dp[i-2]
    dp[i] %= MOD

print(2 * (dp[H] + dp[W] - 1) % MOD)
