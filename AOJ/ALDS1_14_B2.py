# -*- coding: utf-8 -*-

"""
参考：http://perogram.hateblo.jp/entry/2019/05/21/235639
　　　http://drken1215.hatenablog.com/entry/2019/09/16/014600
・ローリングハッシュ
・より汎用的に使えるようにクラス化した版
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
INF = float('inf')
MOD = 10 ** 9 + 7

T = input()
P = input()

class RollingHash:

    MOD = 2 ** 64
    b = 10 ** 8 + 7

    def __init__(self, S):
        # 各文字を数値に変換しておく(0: 48, A: 65, a: 97 から-48する)
        S = [ord(s)-48 for s in S]

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

    def get(self, l, r):
        """ 区間[i,j)のハッシュ値を取得 """
        return (self.hash[r] - self.hash[l] * self.pow[r-l]) % self.MOD

rh_t = RollingHash(T)
rh_p = RollingHash(P)
N = len(T)
M = len(P)

ans = []
p_hash = rh_p.get(0, M)
for i in range(N-M+1):
    # Tの区間[i,i+M)のハッシュ値を取得
    t_hash = rh_t.get(i, i+M)
    if t_hash == p_hash:
        ans.append(i)

if len(ans):
    [print(i) for i in ans]
