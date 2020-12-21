# -*- coding: utf-8 -*-

"""
参考：https://simezi-tan.hatenadiary.org/entry/20140918/1410996988
　　　http://mayokoex.hatenablog.com/entry/2016/06/03/000353
・蟻本演習2-3-11
・木DP、組み合わせ
・辺を描く順番を織り込む時に、全体からグループ毎に選び取っていく、みたいなことをする。
・計算量1000^2=100万にpythonTLEでpypyAC0.8秒。
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

class FactInvMOD:
    """ 階乗たくさん使う時用のテーブル準備 """

    def __init__(self, MAX, MOD):
        """ MAX：階乗に使う数値の最大以上まで作る """
        
        MAX += 1
        self.MAX = MAX
        self.MOD = MOD
        # 階乗テーブル
        factorial = [1] * MAX
        factorial[0] = factorial[1] = 1
        for i in range(2, MAX):
            factorial[i] = factorial[i-1] * i % MOD
        # 階乗の逆元テーブル
        inverse = [1] * MAX
        # powに第三引数入れると冪乗のmod付計算を高速にやってくれる
        inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
        for i in range(MAX-2, 0, -1):
            # 最後から戻っていくこのループならMAX回powするより処理が速い
            inverse[i] = inverse[i+1] * (i+1) % MOD
        self.fact = factorial
        self.inv = inverse
    
    def nCr(self, n, r):
        """ 組み合わせの数 (必要な階乗と逆元のテーブルを事前に作っておく) """

        if n < r: return 0
        # 10C7 = 10C3
        r = min(r, n-r)
        # 分子の計算
        numerator = self.fact[n]
        # 分母の計算
        denominator = self.inv[r] * self.inv[n-r] % self.MOD
        return numerator * denominator % self.MOD

    def nPr(self, n, r):
        """ 順列 """

        if n < r: return 0
        return self.fact[n] * self.inv[n-r] % self.MOD

    def nHr(self, n, r):
        """ 重複組み合わせ """

        # r個選ぶところにN-1個の仕切りを入れる
        return self.nCr(r+n-1, r)

def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

N = INT()
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

def rec(u, prev):
    # 自分の子(子孫分も含む)が持つ辺の数
    edges = []
    for v in nodes[u]:
        if v != prev:
            edges.append(rec(v, u))
            dp[u] *= dp[v]
            dp[u] %= MOD
    # 辺の合計(ここは自分が選ぶ順番を自由に選択できる)
    sm = n = sum(edges)
    for r in edges:
        # 全体nからrずつ選び取っていく
        dp[u] *= fim.nCr(n, r)
        dp[u] %= MOD
        n -= r
    # 辺の合計に自分〜親間の1つを足して返却
    return sm + 1

fim = FactInvMOD(N, MOD)
ans = 0
for i in range(N):
    # 各頂点を出発点として通り数を数える
    dp = [1] * N
    rec(i, -1)
    ans += dp[i]
    ans %= MOD
# 真逆の向きに進んだ分を重複して持っているので/2する
print(fermat(ans, 2, MOD))
