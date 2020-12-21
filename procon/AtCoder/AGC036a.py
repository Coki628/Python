# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/agc036/editorial.pdf
　　　https://www.youtube.com/watch?v=eFQGwwdAVWg
　　　https://atarimae.biz/archives/23716
・数学系
・まずベクトルの外積から、S=x2*y3-x3*y2 (外積の大きさ/2は、2ベクトルの作る三角形の面積になる)
　で、x2,y2を固定して、S=10^9*y3-x3
　あとはキワの処理に気を付けつつ、S/10^9の切り上げと余りでx3,y3を出す。
・具体的な座標がどうなるかは、動画のrngさんが書いたやつが分かりやすい。
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

S=INT()
MAX=10**9

x1, y1 = 0, 0
x2, y2 = MAX, 1
# 割り切れる時は場合分けする
x3 = 0 if S%MAX==0 else MAX-S%MAX
y3 = ceil(S, MAX)

print(x1,y1,x2,y2,x3,y3)
