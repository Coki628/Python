# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/74224
・ほぼ写経from公式解説
・BIT2つ使う。強い人には典型みたい。
・公式解説読み込んで、色々自分で動かして、やっと理解した。
・まず、a,b間の距離がどうなるかについては、a,bの大小関係によって、
　aとbはいずれどこかで交わるので距離は0 or 交わらないのでt=0が最善、の2択に帰着する。
・で、具体的には ai < bi かつ aj <= bj の場合のみが後者、他は前者となる。
・距離0なら総和に影響はないので、ai < bi かつ aj <= bj (t=0) の場合のみを考えれば良いことになる。
　結果的に、t=0位置の座標しか使わないので、位置関係さえ分かれば具体的な速度がどうかはもう使わない。(まじか、ってなった)
・これをまとめるために、aでソート、今見ている要素を右端として、
　自分よりbについて左にある要素だけを左端としてまとめ上げる。
・BIT2本をうまく使うと、累積和っぽく、そのaを右端とした、全ての左端との距離がまとめて取得できる。
　(イメージ図を自分ノートに書いたのでそれを参照)
・取得が終わった後は、次回以降自分も左端候補となるため、BITに自分の情報を詰める。(これはBITをでよくやるやつ)
・個数を持っておくのはBITでよくやったけど、値も持っておけば区間和取得みたいなことができるのは画期的だった。
・計算量は20万とlogで、pypyAC1.7秒。BIT2本使うからか、思ったより結構重い。
"""

import sys
from collections import Counter
from bisect import bisect_left

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

class BIT:

    def __init__(self, n):
        n += 1
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

    def update(self, i, x):
        self.add(i, x - self.get(i))

N = INT()
A = LIST()
B = LIST()

AB = list(zip(A, B))
AB.sort()

B = sorted(set(B))
M = len(B)

# bit1[i] := 速度iの要素の座標の和
bit1 = BIT(M)
# bit2[i] := 速度iの要素数
bit2 = BIT(M)
ans = 0
for a, b in AB:
    # 速度について自分の位置を取得
    pos = bisect_left(B, b)
    # 自分より速度的に左にある要素数と総和を取得
    sm = bit1.sum(pos)
    cnt = bit2.sum(pos)
    # 上記で取得した要素は ai < bi かつ aj <= bj を満たすbなので、これらを全てaとペアにすることを考える
    # この要素数と総和を使うと、累積和の 右端 - 左端 みたいなことができる
    ans += cnt * a - sm
    bit1.add(pos, a)
    bit2.add(pos, 1)
print(ans)
