# -*- coding: utf-8 -*-

"""
・自力AC
・BITにぶたん
・二分探索メソッドの名前が間違ってたりで時間ロスした。
　まあ、おかげでライブラリのバグ取りはできた。
・実際はまあ累積和とかでいけたらしいけど、ちゃんと通ったからOK。
・制約10万にlog2つだけど、pypyAC0.3秒。
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

    def __init__(self, n):
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.tree = [0] * nv

    def sum(self, i):
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s

    def add(self, i, x):
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

    def get(self, l, r=None):
        if r is None: r = l + 1
        res = 0
        if r: res += self.sum(r-1)
        if l: res -= self.sum(l-1)
        return res

    def bisearch_fore(self, l, r, x):
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
            return INF

    def bisearch_back(self, l, r, x):
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
            return -INF

for _ in range(INT()):
    N, K = MAP()
    A = LIST()
    
    if sum(A) <= K:
        print(0)
        continue

    # 数列AをBITに格納
    bit = BIT(N)
    for i, a in enumerate(A):
        bit.add(i, a)
    mx = 0
    for i in range(N):
        # 今回取り除く値
        bit.add(i, -A[i])
        # BIT上の二分探索
        idx = bit.bisearch_fore(0, N-1, K+1)
        # 最大値更新
        if idx > mx:
            mx = idx
            ans = i + 1
        # 今回の値を元に戻す
        bit.add(i, A[i])
    print(ans)
