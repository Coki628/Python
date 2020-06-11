"""
参考：https://www.hamayanhamayan.com/entry/2020/06/07/124554
・自力ならず。
・数列をうまく管理するやつ。
・クエリ回分ソートが必要に見えるが、値の変化は隣接スワップのみなので、実はさほど多くの値の変化がない。
・隣接要素の転倒状態だけBITで管理して、ソートクエリの時は転倒状態がなくなるまでスワップ処理する。
・スワップ処理した際、前後1つずつは大小関係も変化しうるので、常にチェックしてBITも合わせて更新。
・計算量が20万クエリにBITにぶたんでlog2つ、pypyAC0.8秒。やっぱBITのlog軽いわ。
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
    
    def print(self, N):
        for i in range(N):
            print(self.get(i), end=' ')
        print()

def check(x):
    if A[x] > A[x+1] and bit.get(x) == 0:
        bit.add(x, 1)
    if A[x] < A[x+1] and bit.get(x) == 1:
        bit.add(x, -1)

N, Q = MAP()
bit = BIT(N+1)
A = list(range(N+2))
for _ in range(Q):
    t, x, y = MAP()

    if t == 1:
        A[x], A[x+1] = A[x+1], A[x]
        check(x)
        check(x-1)
        check(x+1)
    else:
        while 1:
            idx = bit.bisearch_fore(x, y-1, 1)
            if idx == INF:
                break
            A[idx], A[idx+1] = A[idx+1], A[idx]
            check(idx)
            check(idx-1)
            check(idx+1)
print(*A[1:-1])
