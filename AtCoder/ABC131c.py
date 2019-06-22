# -*- coding: utf-8 -*-

"""
・300点自力AC
・絶対今回こっちのが400点より考えると思う。
・ちょうど最近包除のこと調べてたのもあって、運よくすぐ分かったけど。
・全体-部分1-部分2+重複っていうのは2次元累積和の区間和みたいだしね。
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd

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

def lcm(x, y): return (x * y) // gcd(x, y)

A,B,C,D=MAP()

cntC=B//C-(A-1)//C
cntD=B//D-(A-1)//D
CD=lcm(C, D)
cntCD=B//CD-(A-1)//CD
AB=B-A+1
print(AB-cntC-cntD+cntCD)
