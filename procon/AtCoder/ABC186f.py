"""
参考：https://atcoder.jp/contests/abc186/editorial
・自力ならず。。
・平面走査っぽくやる。自力でも基本的にはその方向で進めていたけど、余事象の「辿り着けないマス」
　の方を数えようとしたのが方針ミスってたっぽい。多分、かえって数えにくくなってた。
　1行、1列のコーナーケースが邪悪で、これがうまく処理できなくて詰んだ。
　辿り着けるマスを数える方針のがまだできたけど、それでも自力で処理できたかは微妙なくらいには十分邪悪。
・とりあえず片方向からを数えて、反対方向から来る方は重複しない分だけ数える。
　どっちの向きも、1が来てしまった時点で打ち切る。(ここ重要)
・重複しない部分の管理には、BITとかで自分の行より上でどこの列に障害物があるかを管理するといい。
　ここでも1の時だけ処理が違う。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**6)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class BIT:
    """ Binary Indexed Tree """

    def __init__(self, n):
        self.n = n
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

    def query(self, l, r):
        """ 区間和の取得 [l, r) """
        if l > r: return 0
        return self.sum(r-1) - self.sum(l-1)
    
    def get(self, i):
        """ 一点取得 """
        return self.query(i, i+1)

    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        self.add(i, x - self.get(i))

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

H, W, M = MAP()
HW = []
adjli1 = [[] for i in range(H+1)]
adjli2 = [[] for i in range(W+1)]
for i in range(M):
    h, w = MAP()
    adjli1[h].append(w)
    adjli2[w].append(h)

# 各列について、右→下の順で進める場所を数える
ans = 0
for w in range(1, W+1):
    if adjli2[w]:
        mnh = min(adjli2[w])
        # 1行目が埋まっていたらもうこの先は無理
        if mnh == 1:
            break
        ans += mnh - 1
    else:
        ans += H

# 各行について、下→右の順で進める場所を、右→下で来れないところだけ数える
bit = BIT(W+1)
for h in range(1, H+1):
    if adjli1[h]:
        mnw = min(adjli1[h])
        # 1列目が埋まっていたらもうこの先は無理
        if mnw == 1:
            break
        if h == 1:
            # 1行目だけは、最初の障害物以降全部が右→下って行けないので、全部埋める
            for w in range(mnw, W+1):
                bit.add(w, 1)
        else:
            for w in adjli1[h]:
                if bit.get(w) == 0:
                    bit.add(w, 1)
        # この行の最初の障害物より手前でかつ、上の行で障害物が出現済のマスが該当する
        ans += bit.query(0, mnw)
    else:
        ans += bit.query(0, W+1)
print(ans)
