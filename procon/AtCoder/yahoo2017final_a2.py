# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/yahoo-procon2017-final/editorial.pdf
　　　http://tutuz.hateblo.jp/entry/2018/07/01/110356
　　　https://ja.wikipedia.org/wiki/%E3%83%AC%E3%83%BC%E3%83%99%E3%83%B3%E3%82%B7%E3%83%A5%E3%82%BF%E3%82%A4%E3%83%B3%E8%B7%9D%E9%9B%A2
・文字列の編集距離(レーベンシュタイン距離)
・LCSみたいな2方向DP
・今回はmod 5で循環させる。
・10万文字*yahoo2回=100万ループはいける気がしたけどpythonTLEでpypyAC0.4秒。
・やっぱりDPは400点でも自力AC大変だなー…。
　まあこれは知らなかったししょうがないけど。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

S=input()
N=len(S)
T='yahoo'
M=len(T)

# dp[i][j] := S[1..i]をyahooyahooyahooyah..(最後は"yahoo"の先頭j文字)にするのにかかる最小操作数
dp = [[INF] * M for i in range(N+1)]
# 初期化(Sが0文字なら、必要な文字数分の操作)
for i in range(M):
    dp[0][i] = i
for i in range(N+1):
    # 循環するので5文字を2周させる
    for j in range(M*2):
        # 挿入
        if i < N:
            dp[i+1][j%5] = min(dp[i+1][j%5], dp[i][j%5]+1)
        # 削除
        dp[i][(j+1)%5] = min(dp[i][(j+1)%5], dp[i][j%5]+1)
        # 置換
        if i < N and S[i] == T[j%5]:
            # 今回増やす文字が一致するなら、ノーコストで遷移できる
            dp[i+1][(j+1)%5] = min(dp[i+1][(j+1)%5], dp[i][j%5])
        elif i < N:
            dp[i+1][(j+1)%5] = min(dp[i+1][(j+1)%5], dp[i][j%5]+1)
# SがN文字あって、mod 5の0(..yahoo)で終わる場合の最小回数
print(dp[N][0])
