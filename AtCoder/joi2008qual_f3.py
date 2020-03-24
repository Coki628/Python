# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2018/01/16/222843
　　　http://dohatsutsu.hatenablog.com/entry/2016/10/18/210317
・分割数の応用
・ちょうどいいから蟻本演習で放置してたこっちの方針もやった。
・頭いい。遷移の詳細は上記参考記事とかソース内コメントにて。
・ざっくり言うと、前の状態から、全部+1ずつしたやつを持ってきて
　いるやつといらないやつを足し引き調整した感じ。
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
MOD = 100000

N, M, S = MAP()

NN = N ** 2
# 1からMまでの整数の中でi種類の数字を選び、合計をjにする場合の数
dp = list2d(NN+1, S+1, 0)
dp[0][0] = 1
for i in range(1, NN+1):
    for j in range(S+1):
        if j-i >= 0:
            # 1からMまでの整数で、i-1種類の数字を選んで合計をj-iにする場合の数(1を含むグループ)
            dp[i][j] += dp[i-1][j-i]
            # 1からMまでの整数で、i種類の数字を選んで合計をj-iにする場合の数(1を含まないグループ)
            dp[i][j] += dp[i][j-i]
        if j-M-1 >= 0:
            # 1からMまでの整数で、i-1種類の数字を選んで合計をj-M-1にする場合の数(Mを超える数が入ってしまうので除外する分)
            dp[i][j] -= dp[i-1][j-M-1]
        dp[i][j] %= MOD
ans = dp[NN][S]
print(ans)
