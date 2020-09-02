"""
・自力ならず。むずかった。。
・完全二分木、木上の数え上げ、LCA
・まず誤読して難しくしてた。条件1を「すべてのi,jについて」って書いてあるのに、
　自分の部分木の頂点が全部自分より大きいみたいな話だと思ってた。。
　でもこの誤読を解消しても全然考察進まなかった。。
・詳細はソース内コメントに記載した。
・LCAを全探索でパス長が一意に、ってのはキーだったな。全然思わなかったけど。。。
"""

import sys
from math import log2

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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

class ModTools:
    """ 階乗・逆元用のテーブルを構築する """

    def __init__(self, MAX, MOD):

        # nCrならn、nHrならn+rまで作る
        MAX += 1
        self.MAX = MAX
        self.MOD = MOD
        factorial = [1] * MAX
        factorial[0] = factorial[1] = 1
        for i in range(2, MAX):
            factorial[i] = factorial[i-1] * i % MOD
        inverse = [1] * MAX
        inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
        for i in range(MAX-2, -1, -1):
            inverse[i] = inverse[i+1] * (i+1) % MOD
        self.fact = factorial
        self.inv = inverse

D, L, R, K = MAP()

# vcnt[d] := 深さdにある頂点の総数
vcnt = [0] * (D+1)
for d in range(1, D+1):
    vcnt[d] = 2**(d-1)
mt = ModTools(vcnt[D], MOD)

# とりあえず番号L,Rを書き込む頂点の深さは一意なので出す
ld = int(log2(L)) + 1
rd = int(log2(R)) + 1
# 2頂点の深さとLCAの深さが分かるとパス長が一意になるので、LCAの深さを全探索
for lcad in range(1, min(ld, rd)+1):
    k = (ld - lcad) + (rd - lcad)
    # パス長がKと一致したら、ここを答えに使う
    if k == K:
        # LCAになりうる頂点数
        lcacnt = vcnt[lcad]
        # 番号Lを書き込める頂点数
        lcnt = vcnt[ld-lcad]
        # LCAとLの頂点が同じ深さなのはコーナーケース処理
        if lcad == ld:
            lcnt = lcacnt
            lcacnt = 1
        # 番号Rを書き込める頂点数
        rcnt = vcnt[rd-lcad]
        # 上記3つの通り数 * 2(左右逆)
        ans = lcacnt * lcnt * rcnt * 2 % MOD
        # L,R以外の番号の振り方を各深さで確認
        for d in range(1, D+1):
            others = vcnt[d]
            # L,Rの分は引く
            if d == ld:
                others -= 1
            if d == rd:
                others -= 1
            # 自由に並べられるので階乗する
            ans *= mt.fact[others]
            ans %= MOD
        print(ans)
        break
else:
    print(0)
