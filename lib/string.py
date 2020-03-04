# -*- coding: utf-8 -*-

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
            self.rank[i] = ord(self.S[i]) if i < self.N else -1
    
        # k文字についてソートされているところから、　2k文字でソートする
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

        # 区間取得のためSparseTableを構築
        self.st = SparseTable(self.lcp, min)

    def getLCP(self, i, j):
        """ S[i:]とS[j:]の共通文字数を取得 """

        return self.st.get(min(self.rsa[i], self.rsa[j]), max(self.rsa[i], self.rsa[j]))


class RollingHash:

    MOD = 2 ** 64
    b = 10 ** 8 + 7

    def __init__(self, S):
        # 各文字を数値に変換しておく
        S = [ord(s)-97 for s in S]

        self.len = len(S)
        self.pow = [1] * (self.len+1)
        for i in range(self.len):
            self.pow[i+1] = self.pow[i] * self.b
            self.pow[i+1] %= self.MOD
        # ハッシュの生成
        self.hash = [0] * (self.len+1)
        for i in range(self.len):
            self.hash[i+1] = self.hash[i] * self.b + S[i]
            self.hash[i+1] %= self.MOD
    
    # 区間[i,j)のハッシュ値を取得
    def get(self, l, r):
        return (self.hash[r] - self.hash[l] * self.pow[r-l]) % self.MOD


def Z_algorithm(S):
    N = len(S)
    res = [0] * N
    res[0] = N
    i = 1
    j = 0
    while i < N:
        while i+j < N and S[j] == S[i+j]:
            j += 1
        res[i] = j
        if j == 0:
            i += 1
            continue
        k = 1
        while i+k < N and k+res[k] < j:
            res[i+k] = res[k]
            k += 1
        i += k
        j -= k
    return res

def KMP(S: str, T: str) -> list:
    """ KMP法(Sの中にTとマッチする場所のindexのリストを返す)：O(|S|+|T|) """

    N = len(S)
    M = len(T)
    # KMPをやるための前計算テーブルを構築
    table = [0] * (M+1)
    table[0] = -1
    j = -1
    for i in range(M):
        while j >= 0 and T[i] != T[j]:
            j = table[j]
        j += 1
        table[i+1] = j
    # KMP法
    res = []
    m = i = 0
    while m+i < N:
        if T[i] == S[m+i]:
            i += 1
            if i == M:
                res.append(m)
                m = m + i - table[i]
                i = table[i]
        else:
            m = m + i - table[i]
            if i > 0:
                i = table[i]
    return res
