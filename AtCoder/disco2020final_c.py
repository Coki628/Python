# -*- coding: utf-8 -*-

"""
・せっかく作ったSuffix Arrayを使ってみようと思って挑むも撃沈TLE。。
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

class SuffixArray:

    class SparseTable:

        def __init__(self, A, func):
            self.N = len(A)
            self.func = func
            h = 0
            while 1<<h < self.N:
                h += 1
            self.dat = list2d(h, 1<<h, 0)
            self.height = [0] * (self.N+1)

            for i in range(2, self.N+1):
                self.height[i] = self.height[i>>1] + 1
            for i in range(self.N):
                self.dat[0][i] = A[i]
            for i in range(1, h):
                for j in range(self.N):
                    self.dat[i][j] = self.func(self.dat[i-1][j], self.dat[i-1][min(j+(1<<(i-1)), self.N-1)])
            
        def get(self, l, r):
            """ 区間[l,r)でのmin,maxを取得 """

            if l >= r:
                raise Exception
            a = self.height[r-l]
            return self.func(self.dat[a][l], self.dat[a][r-(1<<a)])

    def __init__(self, S):
        self.k = 1
        if S and type(S[0]) == str:
            self.S = [ord(s) for s in S]
        else:
            self.S = S
        self.N = len(S)
        self.rank = [0] * (self.N+1)
        self.tmp = [0] * (self.N+1)
        self.sa = [0] * (self.N+1)
        self.build_sa()

    # def compare_sa(self, i, j):
    #     if self.rank[i] != self.rank[j]:
    #         return self.rank[i] < self.rank[j]
    #     else:
    #         ri = self.rank[i+self.k] if i + self.k <= self.N else -1
    #         rj = self.rank[j+self.k] if j + self.k <= self.N else -1
    #         return ri < rj

    def compare_sa(self, i, j):
        """ saの比較関数 """

        # 第1キー
        if self.rank[i] == self.rank[j]: res1 =  0
        elif self.rank[i] < self.rank[j]: res1 =  -1
        else: res1 = 1
        # 第2キー
        ri = self.rank[i+self.k] if i + self.k <= self.N else -1
        rj = self.rank[j+self.k] if j + self.k <= self.N else -1
        if ri == rj: res2 = 0
        elif ri < rj: res2 = -1
        else: res2 = 1
        # 比較
        return res1 or res2

    def build_sa(self):
        """ Suffix Arrayの構築 """
        from functools import cmp_to_key

        # 最初は1文字、ランクは文字コード
        for i in range(self.N+1):
            self.sa[i] = i
            self.rank[i] = self.S[i] if i < self.N else -1
    
        # k文字についてソートされているところから、2k文字でソートする
        k = 1
        while k <= self.N:
            self.k = k
            self.sa = sorted(self.sa, key=cmp_to_key(self.compare_sa))
            k *= 2

            # いったんtmpに次のランクを計算し、それからrankに移す
            self.tmp[self.sa[0]] = 0
            for i in range(1, self.N+1):
                self.tmp[self.sa[i]] = self.tmp[self.sa[i-1]] + (1 if self.compare_sa(self.sa[i-1], self.sa[i]) else 0)
            for i in range(self.N+1):
                self.rank[i] = self.tmp[i]

        # 文字列が陽に持てる量ならこっちのが速い
        # from operator import itemgetter

        # suffix_arr = [(self.S[i:], i) for i in range(self.N+1)]
        # suffix_arr.sort(key=itemgetter(0))
        # for i in range(self.N+1):
        #     self.sa[i] = suffix_arr[i][1]

    def build_lcp(self):
        """ LCP配列の構築 """

        self.lcp = [0] * (self.N+1)
        self.rsa = [0] * (self.N+1)
        for i in range(self.N+1):
            self.rsa[self.sa[i]] = i
        h = 0
        self.lcp[0] = 0
        for i in range(self.N):
            # 文字列内での位置iの接尾辞と、接尾辞配列内でその1つ前の接尾辞のLCPを求める
            j = self.sa[self.rsa[i]-1]

            # hを先頭の分1減らし、後ろが一致しているだけ増やす
            if h > 0:
                h -= 1
            while j+h < self.N and i+h < self.N:
                if self.S[j+h] != self.S[i+h]:
                    break
                h += 1
            self.lcp[self.rsa[i]-1] = h

    def build_st(self):
        """ 区間取得のためSparseTableを構築 """

        self.st = self.SparseTable(self.lcp, min)

    def getLCP(self, i, j):
        """ S[i:]とS[j:]の共通文字数を取得 """

        return self.st.get(min(self.rsa[i], self.rsa[j]), max(self.rsa[i], self.rsa[j]))

class BIT:
    """ Binary Indexed Tree """

    def __init__(self, n):
        # 0-indexed
        n += 1
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

    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        self.add(i, x - self.get(i))

    def bisearch_fore(self, l, r, x):
        """ 区間[l,r]を左から右に向かってx番目の値がある位置 """
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

N = INT()
A = LIST()

sa = SuffixArray(A)

ans = [0] * N
bit = BIT(N+1)
bit.add(N, 1)
for idx in sa.sa[1:]:
    # 自分より大きい最小のindex
    res = bit.bisearch_fore(idx+1, N, 1)
    ans[idx] = res
    bit.add(idx, 1)
for a in ans: print(a)
