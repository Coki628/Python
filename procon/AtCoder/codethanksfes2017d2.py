# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/code-thanks-festival-2017-open/editorial.pdf
　　　http://tutuz.hateblo.jp/entry/2018/07/15/171604
・式変形、合同式、詳細は参考にて。
・これを自分で導き出す数学系の考察力はどうやって身に付けようか。
・ちなみにgcd(A,B)の計算量はO(logMAX(A,B))とのこと。
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

N,M=MAP()

print(M-gcd(N, M))
