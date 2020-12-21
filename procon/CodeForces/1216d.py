# -*- coding: utf-8 -*-

"""
・自力AC
・GCD系
・新しい知見として、こどふぉ環境だとfractionsのGCDが遅くてmathのGCD使った方がいいっぽい。
　内部的に同じもの呼んでると思ってたんだけど、最初10万のNで2秒をTLEして、mathの方にしたら通った。
・あと余談だけど、これの途中であーちゃん泣いて30分くらい離席してたから、本当はもう30分くらい早く解けたはず。
　まあ30分増えたところでEは解けなかったし、そんなに影響なかったとは思うけど。
"""

import sys
from math import gcd

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

N = INT()
A = LIST()

mn = min(A)
A = [a-mn for a in A]
g = 0
for a in A:
    g = gcd(g, a)

if g == 0:
    print(0, 0)
    exit()

mx = max(A)
sm = 0
for a in A:
    sm += (mx - a) // g
print(sm, g)
