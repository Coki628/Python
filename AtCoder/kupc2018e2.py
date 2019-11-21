# -*- coding: utf-8 -*-

"""
参考：http://www.kupc.jp/static/media/E.267336ab.pdf
　　　https://kimiyuki.net/writeup/algo/atcoder/kupc2018-e/
　　　https://atcoder.jp/contests/kupc2018/submissions/3309665
・長らく保留にしてたやつ、やっと噛み砕いて通した。
・転倒数数え上げ、順列、辞書順、BIT、桁DP
・順列の転倒数の総和。まずこれ知らないで導けるのすごいなぁと思う。
・辞書順でP以下しか使えないので、桁毎に値が確定したらそれ以下を決めていく感じ。
・各桁が決まる毎に3つのグループを足していく。(内容はコメントと自分ノート参照)
・計算量は20万とlogで約360万、pypyAC0.9秒。
・他の人よりやや遅かったので速度改善。
　演算が全部掛け算なので、+=する前にも常にmod取るようにした。
・これでpypyAC0.6秒。
・ちなみに事前準備で試しに今回使わない逆元作らないようにもしてみたけど速度変化なし。
・桁DPっぽいという話もあったので、分かりやすく桁DPっぽくしてみた版。
・DP配列持つ分、まあ不利だとは思ったけど、やっぱりちょっと遅くなってpypyAC0.8秒。
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

class ModTools:
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
    
    def div(self, x, y):
        """ MOD除算 """

        return x * pow(y, self.MOD-2, self.MOD) % self.MOD

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

N = INT()
P = LIST()
mt = ModTools(N, MOD)

# 長さnの順列の転倒数の総和
def calc(n):
    return mt.div(mt.fact[n], 2) * (n*(n-1)//2) % MOD

# dp[i][j] := i桁目まで見て、作る数がPと等しい(j=0)か小さい(j=1)場合の転倒数の総和
dp = list2d(N+1, 2, 0)
bit = BIT(N+1)
for i, p in enumerate(P):
    # pより右に小さい値が現れる数(現在位置で辞書順をPより小さくするのに使える数)
    cur = p - bit.sum(p) - 1
    # Pと等しい場合の総和に今回分を足す
    dp[i+1][0] = dp[i][0] + cur
    dp[i+1][0] %= MOD
    # Pより小さい数の転倒数総和を数える
    cnt = 0
    # 今回位置で使える小さい値の数 * 今後決める部分の転倒数総和
    cnt += cur * calc(N-1-i) % MOD
    cnt %= MOD
    # 今回決める部分と今後決める部分をまたぐ転倒数 * 今後決める部分の全通り
    cnt += cur*(cur-1)//2 * mt.fact[N-1-i] % MOD
    cnt %= MOD
    # 既に決まっている部分の転倒数総和 * 今回位置で使える小さい値の数 * 今後決める部分の全通り
    cnt += dp[i][0] * cur * mt.fact[N-1-i] % MOD
    cnt %= MOD
    dp[i+1][1] = dp[i][1] + cnt
    dp[i+1][1] %= MOD
    # pを出現済とする
    bit.add(p, 1)
print((dp[N][0]+dp[N][1])%MOD)
