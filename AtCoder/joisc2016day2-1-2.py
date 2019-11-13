# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/camp/2016/2016-sp-tasks/2016-sp-d2-employment-review.pdf
・蟻本演習3-3-3
・区間加算BIT(区間加算・区間合計取得)、座標圧縮
・前半パートの処理はjoi2018qual_d(日本沈没)と同じような操作でいける。
・20万クエリでpythonTLE,pypyAC2.0秒(制約5秒)
・解説を参考に、初期値もクエリとみなした版
・実装はこっちのが全然軽い。(上記前半パートいらない)
・速度もちょっと速くなって、pypyAC1.9秒。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

class BIT2:
    """ 区間加算BIT(区間加算・区間合計取得) """

    def __init__(self, N):
        # 添字0が使えないので、内部的には全て1-indexedとして扱う
        N += 1
        self.N = N
        self.data0 = [0] * N
        self.data1 = [0] * N

    def _add(self, data, k, x):
        k += 1
        while k < self.N:
            data[k] += x
            k += k & -k

    def _get(self, data, k):
        k += 1
        s = 0
        while k:
            s += data[k]
            k -= k & -k
        return s

    def add(self, l, r, x):
        """ 区間[l,r)に値xを追加 """

        self._add(self.data0, l, -x*(l-1))
        self._add(self.data0, r, x*(r-1))
        self._add(self.data1, l, x)
        self._add(self.data1, r, -x)

    def query(self, l, r):
        """ 区間[l,r)の和を取得 """

        return self._get(self.data1, r-1) * (r-1) + self._get(self.data0, r-1) \
             - self._get(self.data1, l-1) * (l-1) - self._get(self.data0, l-1)

def compress(A):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(set(A))):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N, M = MAP()

A = LIST(N)

A2 = []
Q = []
# 初期値もクエリとみなす
for i, a in enumerate(A):
    Q.append((2, i+1, a))

# クエリ先読み
for i in range(M):
    t, *arg = MAP()
    if t == 1:
        b, = arg
        A2.append(b)
        Q.append((t, b))
    
    else:
        c, d = arg
        A2.append(d)
        Q.append((t, c, d))

# 座標圧縮
mx = max(max(A), max(A2))
zipped, _ = compress([0, mx+1] + A + A2)

# 以後、圧縮後の値で全て扱う
L = len(zipped)
# 全候補者は評価値の初期値を0とする
A = [0] * (N+2)
# bit[i] := 評価値について、位置iを採用ラインにした場合のグループ数
bit = BIT2(L)

ans = []
for t, *arg in Q:
    if t == 1:
        b, = arg
        b = zipped[b]
        ans.append(str(bit.query(b, b+1)))
    else:
        c, d = arg
        d = zipped[d]
        prevd = A[c]
        mx = max(A[c-1], A[c+1])
        mn = min(A[c-1], A[c+1])
        # 評価が上がる
        if d > prevd:
            # 左右の候補者の評価値から、グループ数の増減を区間で場合分け
            if mn - prevd > 0:
                bit.add(prevd+1, min(mn, d)+1, -1)
            if d - mx > 0:
                bit.add(max(mx, prevd)+1, d+1, 1)
        # 評価が下がる
        elif d < prevd:
            if prevd - mx > 0:
                bit.add(max(mx, d)+1, prevd+1, -1)
            if mn - d > 0:
                bit.add(d+1, min(mn, prevd)+1, 1)
        # 候補者cの評価値を更新
        A[c] = d

print('\n'.join(ans))
