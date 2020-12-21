# -*- coding: utf-8 -*-

"""
参考：ツイート
・自力ならず。。
・論理積ANDの性質
・これもコンテスト中には本質を見抜けなかった。
　問題文のDP実際に試したのに、あれー？最大値出るじゃん、何でだ？とかなってた。(出ません)
　B,C解けなくて相当テンパった精神状態で挑んだから、まともに見れてなかった感はある。。
・DPではmaxで更新しようとしているので、下位ビットがどんなに立っていても上位ビットが立っている方を
　優先して更新する。このことをうまく利用する。
・やり方としては、Kは17ビットの範囲に収まるので、18ビット目だけを立てた場所を用意して、
　DPにはそっちで更新させるように持っていく。最後に17ビット全立ちの場所を作れば、
　DPは0、実際の答えはKになって全てうまくいく。
・具体的にどうやったかは、ソースコード及び自分ノート参照。
・ちゃんと自分で考えて構築したけど、ツイートでヒントを得ていたのはあるので、
　本質に気づいていたとして、ノーヒントで構築できたかは分からないな。。
"""

import sys
from collections import Counter

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

K = INT()

grid = list2d(3, 3, (1<<18)-1)
grid[0][2] = grid[2][0] = 1<<17
grid[1][1] = K
grid[2][2] = (1<<17) - 1

print(3, 3)
for i in range(3):
    print(*grid[i])
