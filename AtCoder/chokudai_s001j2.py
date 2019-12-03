# -*- coding: utf-8 -*-

"""
参考：https://ikatakos.com/pot/programming_algorithm/dynamic_programming/inversion
・BIT、転倒数数え上げ
・ループ逆にした版
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

class BIT:
    def __init__(self, n):
        # 0-indexed
        self.size = n
        self.tree = [0] * (n+1)
 
    def sum(self, i):
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s
    
    # 区間和の取得 [l, r)
    def get(self, l, r=None):
        # 引数が1つなら一点の値を取得
        if r is None: r = l+1
        res = 0
        if r: res += self.sum(r-1)
        if l: res -= self.sum(l-1)
        return res

    # 値の追加：添字, 値
    def add(self, i, x):
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

N = INT()
A = LIST()

bit = BIT(N+1)
ans = 0
for i in range(N-1, -1, -1):
    # 自分より右にある、自分より小さな数の個数
    ans += bit.sum(A[i])
    bit.add(A[i], 1)
print(ans)
