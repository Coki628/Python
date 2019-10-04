# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc137/editorial.pdf
・WAだった自分のやつ、想定解ではないけどループ逆順にするだけでACになった。
　条件の厳しい方からやるとか基本なのに、、もったいない。
・セグメント木で区間最大とindex取得して、その分減らして更新。
・狭い方からやってるので、同値は左優先でも問題なかった。
・計算量は10万の多分NlogN系だけど、やっぱセグ木大変みたいでpypyで1.2秒AC。
・index取得セグ木にも速度改善版を適用してみた。pypy1.0秒AC。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

class SegTreeIndex:
    """
    以下のクエリを処理する
    1.update:  i番目の値をxに更新する
    2.query: 区間[l, r)の値とindex(同値があった場合は一番左)を得る
    """
 
    def __init__(self, n, func, init):
        """
        :param n: 要素数(0-indexed)
        :param func: 値の操作に使う関数(min, max)
        :param init: 要素の初期値(単位元)
        """
        self.n = n
        self.func = func
        self.init = init
        # nより大きい2の冪数
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.init] * (n2 << 1)
        self.index = [self.init] * (n2 << 1)
        # 1段目(最下段)の初期化
        for i in range(n2):
            self.index[i+n2] = i
        # 2段目以降の初期化
        for i in range(n2-1, -1, -1):
            # 全部左の子の値に更新
            self.index[i] = self.index[i*2]

    def update(self, i, x):
        """
        i番目の値をxに更新
        :param i: index(0-indexed)
        :param x: update value
        """
        i += self.n2
        self.tree[i] = x
        while i > 1:
            left, right = min(i, i^1), max(i, i^1)
            if self.func(self.tree[left], self.tree[right]) == self.tree[left]:
                self.tree[i >> 1] = self.tree[left]
                self.index[i >> 1] = self.index[left]
            else:
                self.tree[i >> 1] = self.tree[right]
                self.index[i >> 1] = self.index[right]
            i >>= 1

    def query(self, a, b):
        """
        [a, b)の値を得る
        :param a: index(0-indexed)
        :param b: index(0-indexed)
        """
        l = a + self.n2
        r = b + self.n2
        s = (self.init, -1)
        while l < r:
            if r & 1:
                r -= 1
                res = self.func(s[0], self.tree[r])
                # 左との一致を優先する
                if res == s[0]:
                    pass
                else:
                    s = (self.tree[r], self.index[r])
            if l & 1:
                res = self.func(self.tree[l], s[0])
                # 左との一致を優先する
                if res == self.tree[l]:
                    s = (self.tree[l], self.index[l])
                else:
                    pass
                l += 1
            l >>= 1
            r >>= 1
        return s

N, M = MAP()
# 日数別に報酬の値を格納
days = [[] for i in range(10**5+1)]
for i in range(N):
    a, b = MAP()
    days[a].append(b)

# indexに日数、値に報酬の値を入れるセグ木
sti = SegTreeIndex(10**5+1, max, -INF)
for i in range(10**5+1):
    # 報酬の値をソート
    days[i].sort()
    if len(days[i]):
        # その日数での最大値でセグ木を更新
        sti.update(i, days[i][-1])

ans = 0
# 制約の厳しい後半から確認していく
for i in range(M-1, -1, -1):
    # 現在日iで間に合う範囲にある報酬の最大値とindexを取得
    val, idx = sti.query(0, M-i+1)
    if val == -INF:
        continue
    ans += val
    # 日数別のリストから今回分を削除
    days[idx].pop(-1)
    # それに合わせてセグ木も更新
    if len(days[idx]):
        sti.update(idx, days[idx][-1])
    else:
        sti.update(idx, -INF)
print(ans)
