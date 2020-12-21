# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC
・Bだけど難易度緑なので復習。
・ようは区間和取る時みたいなことをやるんだけど、
　B問題としては考察難度最強クラスかなーって感じ。
・Bにまともに挑む頃だと絶対愚直でやる考えになるもんね。。
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

a, b, x = MAP()

# 区間和っぽくやる
l = (a-1) // x
r = b // x
print(r - l)
