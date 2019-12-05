# -*- coding: utf-8 -*-

"""
・自力WA
・転倒数数え上げ応用、ある転倒数の順列の作成
・多分これだと、転倒数は合ってても、A,B両者の間の数列とは限らない。
・転倒数が同じでも、例えば1,3,2と2,1,3みたいに色んな並びがある。(考えてみればそれはそう)
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

class BIT:
    """ Binary Indexed Tree """

    def __init__(self, n):
        # 0-indexed
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.tree = [0] * nv

    def sum(self, i):
        """ [0, i]を合計する """
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s

    def add(self, i, x):
        """ 値の追加：添字i, 値x """
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

    def get(self, l, r=None):
        """ 区間和の取得 [l, r) """
        # 引数が1つなら一点の値を取得
        if r is None: r = l + 1
        res = 0
        if r: res += self.sum(r-1)
        if l: res -= self.sum(l-1)
        return res

N = INT()
A = [a-1 for a in LIST()]
B = [b-1 for b in LIST()]

atoi = {}
itoa = {}
for i, a in enumerate(A):
    atoi[a] = i
    itoa[i] = a

# 数列Aを順番通り(転倒数0)とみなした時のB
B2 = [atoi[b] for b in B]

# この転倒数がA,B間の転倒距離
bit = BIT(N)
cnt = 0
for i in range(N-1, -1, -1):
    cnt += bit.sum(B2[i])
    bit.add(B2[i], 1)

# ちょうど真ん中を作れない
if cnt % 2 == 1:
    print(-1)
    exit()

# 転倒数targetの順列を作る
target = cnt // 2
C = []
used = set()
idx = N - 1
while target > 0:
    if target < N:
        C.append(target)
        used.add(target)
        break
    else:
        C.append(idx)
        used.add(idx)
        target -= idx
for i in range(N):
    if i not in used:
        C.append(i)

# 元の値に戻す
C2 = [itoa[c]+1 for c in C]
print(*C2)
