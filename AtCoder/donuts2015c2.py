# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/kuno4n/donuts-2015
・公式解
・スタック
・頭いい。言われると、確かにそれでいける、ってなる。
・スタックが想定解のやつだいたいいつも気付けないんだけど、
　どうやったら見抜けるようになるだろうな。。。
・これは10万のO(N)でlogもつかないので、さすがに余裕のpythonAC0.2秒、pypyAC0.3秒。
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

N = INT()
A = LIST()

stack = []
ans = [0] * N
for i, a in enumerate(A):
    # 各要素に到達した時のスタックのサイズが答え
    ans[i] = len(stack)
    # 自分より背の低い情報は、それ以降捨てていい
    while stack and stack[-1] < a:
        stack.pop()
    stack.append(a)
[print(a) for a in ans]
