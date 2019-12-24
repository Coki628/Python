# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc046
　　　https://tokumini.hatenablog.com/entry/2018/09/29/190431
　　　http://mayokoex.hatenablog.com/entry/2015/12/14/125326
・これは考えても出なかった。巧いなぁと思う。
・ソートと多重集合で、異なる2つの条件をうまく管理する。
・男女まとめてソートしておけば、同値で女性(追加)を男性(検索)より先にしておくだけで、一括管理できてスッキリ。
・pythonはmultisetがないので、BIT上の二分探索を活用する。
・計算量、N+M=最大30万で、BITにぶたんはlog2つなので大変。pypyAC2.3秒。(制約3秒)
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

    def bisearch_left(self, l, r, x):
        """ 区間[l,r]を右から左に向かってx番目の値がある位置 """
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

    def bisearch_right(self, l, r, x):
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
            return -1

def compress(S):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N, M = MAP()
AB = []
CD = []
S = set()
for i in range(N):
    a, b = MAP()
    AB.append((a, b))
    S.add(b)
for i in range(M):
    c, d = MAP()
    CD.append((c, d))
    S.add(c)

# BITに乗せるために座標圧縮しておく
zipped, _ = compress(S)
L = len(zipped)

# 男女まとめたリストを作る
li = []
for a, b in AB:
    # (自分の年収, 男性, 希望年収)
    li.append((a, 1, zipped[b]))
for c, d in CD:
    # (希望年収, 女性, 自分の年収)
    li.append((d, 0, zipped[c]))

# 女→男側の年収と希望、同値なら女性先でソート
li.sort()

bit = BIT(L)
ans = 0
for _, g, x in li:
    if g == 0:
        # ここから先で見る男性は、女性の希望年収を満たしているのでこの女性を待ちに追加する
        bit.add(x, 1)
    else:
        # 待ちの集合に入っている女性から、自分の希望を満たす最小の人を選ぶ
        res = bit.bisearch_right(x, L-1, 1)
        if res != -1:
            ans += 1
            bit.add(res, -1)
print(ans)
