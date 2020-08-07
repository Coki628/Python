"""
・結構時間取られたけど自力AC
・BIT上の二分探索
・最初、次の場所を探すのに普通に二分探索していてWA。。
　使える位置は不規則に存在するので普通ににぶたんだとダメ。
・で、愚直にwhileで前から1つずつ見たらTLEだったのでそこから高速化を考える。
・で、0,1それぞれについて、使って大丈夫な場所を保持するBITを持つと、BITにぶたんで探せる。
・20万のNにlog2つだけど、pypyAC0.64秒。
・0,1に対称に同じ処理だったので、パリティ管理して1つにまとめてみた。
　コードはすっきりしたけど、何やってるかの可読性は悪くなった気がするので微妙。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

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

    def bisearch_fore(self, l, r, x):
        """ 区間[l, r]を左から右に向かってx番目の値がある位置 """
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
            return INF

for _ in range(INT()):
    N = INT()
    S = list(map(int, input()))

    idx = [0] * 2
    ans = [0] * N
    ok = [BIT(N+1) for i in range(2)]
    for i in range(N+1):
        ok[0].add(i, 1)
        ok[1].add(i, 1)
    for i, s in enumerate(S):
        t = s ^ 1
        ans[i] = idx[s] + 1
        idx[t] = min(idx[t], idx[s])
        ok[s].update(idx[s], 0)
        ok[t].update(idx[s], 1)
        idx[s] = ok[s].bisearch_fore(idx[s], N, 1)
    print(max(ans))
    print(*ans)
