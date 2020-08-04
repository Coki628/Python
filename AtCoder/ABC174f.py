"""
参考：https://hama-du-competitive.hatenablog.com/entry/2016/10/01/001418
・自力ならず。。
・区間、種類数、クエリ、ソートしてBIT
・この記事の説明が自分的には分かりやすかった。
・右端順に並べたクエリを順番に見ていって、その時の右端に応じて、各値が最後に出現した位置を更新していく。
・理解しないと結構ややこしめな処理な気がするけど、これ水diffはやっぱりググって貼った勢が多いんだろうな。。
・計算量、50万のNとクエリでソートとBITのlogが乗るんだけど、結構きつくてpypyAC1.68秒。
・この問題、制約きつめだけどMo's algorithmでも解けるようなので、いずれそっちもやってみてもいいかも。
・あとWavelet Matrixってやつ使うとクエリ先読みもしなくていけるらしい。すごいね。
"""

import sys
from operator import itemgetter

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

    def print(self, N):
        for i in range(N):
            print(self.get(i), end=' ')
        print()

N, Q = MAP()
A = [a-1 for a in LIST()]

LR = []
for i in range(Q):
    l, r = MAP()
    l -= 1; r -= 1
    LR.append((l, r, i))

# 右端でソート
LR.sort(key=itemgetter(1))
# prev[i] := ある時点までで、色iが最後に出現したindex
prev = [-1] * N
# bit[i] := ある時点までで、位置iに最後に出現した色の種類数
bit = BIT(N)
j = 0
ans = [0] * Q
# 右端基準で処理を進めていく
for l, r, i in LR:
    # 位置rまでの数列Aの要素を確認してprevとBITを更新する
    while j <= r:
        # 前回出現位置があったら、BITの位置prev[A[j]]から色A[j]分のカウントを取り除く
        if prev[A[j]] != -1:
            bit.add(prev[A[j]], -1)
        # 色A[j]の位置をjに更新、BITのカウントも追加する
        prev[A[j]] = j
        bit.add(j, 1)
        j += 1
    # 区間[l,r]に入っている種類数が答え
    ans[i] = bit.get(l, r+1)

for a in ans:
    print(a)
