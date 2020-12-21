# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=fYS-rAUSD5o
　　　https://img.atcoder.jp/arc103/editorial.pdf
・自力ならず。。
・木の構築
・解説動画の図が一番分かりやすかった。
・欲しい要素数でない限り、ある固定の頂点に次を繋ぎ続けて足を増やす。
・欲しい要素数が来たら、固定の頂点を前に動かす。するとその要素数で切れるようになるのでうまくいく。
・自明に無理なやつを除いて、この方法で全て作れる。
・理解したら、確かにー、すげー、って感じだけど、この構築方法どうやって思いつくんだ。。
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

S = input()
N = len(S)
S = '0' + S
 
# 絶対無理なやつを弾いておく
if S[1] == '0' or S[N] == '1' or S != S[::-1]:
    print(-1)
    exit()

edges = []
base = 1
for i in range(1, N):
    # 欲しい要素数でない限り、baseに繋ぎ続けて足を増やす
    edges.append((base, i+1))
    if S[i] == '1':
        base = i + 1
[print(*edge) for edge in  edges]
