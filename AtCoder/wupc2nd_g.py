# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/wupc2nd/submissions/1416728
　　　https://www.slideshare.net/hama_du/g-daruma
・蟻本演習3-3-2
・BIT上の二分探索
・コーナーケースが潰せず自力ならず。同じケースで落ちてる人探して何とか原因判明させてAC。
・計算量は10万にlog2つ、pypyAC1.1秒。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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

    def bisearch_left(self, l, r, x):
        """ 区間[l,r]で左からx番目の値がある位置 """
        l_sm = self.sum(l-1)
        ok = r + 1
        ng = l - 1
        while ng+1 < ok:
            mid = (ok+ng) // 2
            if self.sum(mid) - l_sm >= x:
                ok = mid
            else:
                ng = mid
        if ok != r + 1:
            return ok
        else:
            return -1

    def bisearch_right(self, l, r, x):
        """ 区間[l,r]で右からx番目の値がある位置 """
        r_sm = self.sum(r)
        ok = l - 1
        ng = r + 1
        while ok+1 < ng:
            mid = (ok+ng) // 2
            if r_sm - self.sum(mid-1) >= x:
                ok = mid
            else:
                ng = mid
        if ok != l - 1:
            return ok
        else:
            return -1

N, M, H = MAP()
A = LIST()

bit = BIT(N+M)
for i, a in enumerate(A):
    bit.add(i, a)

for i in range(M):
    opt, arg = input().split()
    arg = int(arg)
    if opt == 'add':
        bit.add(N+i, arg)
    else:
        # ハンマーの下端
        s = bit.bisearch_left(0, N+M-1, arg-H+1)
        # 下端が範囲外(空振り)
        if s == -1:
            print('miss')
            continue
        # ハンマーの上端
        t = bit.bisearch_left(0, N+M-1, arg+H)
        # だるまの上端
        u = bit.bisearch_right(0, N+M-1, 1)
        # 上端だけ範囲外かつ下端とだるまの上端が同位置 or 上端と下端が同位置なら叩ける
        if (t == -1 and s == u) or s == t:
            print('go')
            bit.add(s, -bit.get(s))
        else:
            print('stop')
