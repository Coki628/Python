# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc024
・式変形
・組み合わせの式はそのままだと階乗を含むけど、比の形にすることで
　普通の四則計算になって、あとは連立方程式を解くだけ。
・最終的にプログラムに書く式まで持ってくのにすごく時間がかかった。
　連立方程式を展開して解く時、もっと速くて正確な感じでやれればいいのに。
　(苦労したのは自分ノート参照)
・MODの世界なので割り算はフェルマー使う。
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

def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

A=INT()
B=INT()
C=INT()

# 長い式変形の末、こうなる
a=fermat(B*C-A*C, A*B+A*C-B*C, MOD)
b=fermat(B*C-A*B, A*B+A*C-B*C, MOD)
print(a, b)
