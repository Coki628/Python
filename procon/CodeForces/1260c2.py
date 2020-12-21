# -*- coding: utf-8 -*-

"""
・英語の書き込みとか人のコード見ても、何でそうなるのか理解できなかったが、
　GCDを使うっていうヒントを頼りに後から色々考察し直したら正解に辿り着けた。
・まず、GCDにかけて互いに素にすることで、話を簡単にしておく。(相対的な両者の関係は変化しない)
・次に、大きい方(ここではBとする)が次に来るまでの間の区間に注目。その区間長はB-1になる。
・最悪ケースではこの区間に小さい方(A)が端からぴったり入ったりするので、(例えばA=3,B=5ならAは6,9で2回入る)
　この区間に最大でAがいくつ入るか、が答えになる。
・具体的に式にすると、これは区間長/Aの切り上げなので、ceil(B-1, A)すればいいことが分かる。
・日本語情報少ないし、迷宮入りで終了かなー、と思ったけど、ちゃんと理解して通せて良かった。
"""

import sys
from fractions import gcd

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

for _ in range(INT()):
    r, b, k = MAP()
    # 大小関係を固定
    r, b = min(r, b), max(r, b)
    # GCDで互いに素にして話を簡単にしておく(相対的な両者の関係は変化しない)
    d = gcd(r, b)
    r //= d
    b //= d
    # 次に大きい方が来るまでの区間(b-1)に、小さい方(r)が最大で何回収まるか、がk未満ならOK
    if ceil(b-1, r) < k:
        print('OBEY')
    else:
        print('REBEL')
