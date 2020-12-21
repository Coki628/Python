"""
参考：https://twitter.com/i/events/912020224261500928
　　　https://atcoder.jp/contests/jag2017summer-day1/submissions/13648811
・むずい。久々に再着手したけど、当時のインチキDPから考察は進まなかった。。
　でも当時読んでもあんまり理解できなかった解説がやや分かったので解いておいた。
・BIT上の二分探索、LISに帰着
・最終的な並びを考えると、そこで位置が交差するような2匹を同時にスローロリスにはできない。
　この条件で値を最大化しようとすると、ちょうどLISそのものに帰着できる。(まずこれも気付けない)
・最終的な並びは、BITにぶたんをうまく使うとNとlog2つで作れるので間に合う。(用法はソース内コメントにて)
・とはいえ30万のNにlog2つはちょっと心配だったけど、新環境pypyはさすがのAC0.76秒。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def LIS(A: list, equal=False) -> list:
    """ 最長増加部分列 """
    from operator import gt, ge
    from bisect import bisect_left, bisect_right

    compare = gt if not equal else ge
    bisect = bisect_left if not equal else bisect_right
    L = [A[0]]
    for a in A[1:]:
        if compare(a, L[-1]):
            L.append(a)
        else:
            L[bisect(L, a)] = a
    return L

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

    def bisearch_back(self, l, r, x):
        """ 区間[l, r]を右から左に向かってx番目の値がある位置 """
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
            return -INF

N = INT()
A = LIST()

bit = BIT(N)
for i in range(N):
    # まだ空いているindexに値を入れておく
    bit.add(i, 1)
B = [0] * N
for i in range(N-1, -1, -1):
    # 全体を対象に、後ろからa+1番目の値を取得
    j = bit.bisearch_back(0, N-1, A[i]+1)
    # ここにi番目のやつを入れて、使用済indexのjをBITから取り除く
    B[j] = i
    bit.add(j, -1)

ans = len(LIS(B))
print(ans)
