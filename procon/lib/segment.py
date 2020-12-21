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
            return -1

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
            return -1


class BIT:
    """ Binary Indexed Tree (MOD付) """

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
            s %= MOD
            i -= i & -i
        return s

    def add(self, i, x):
        """ 値の追加：添字i, 値x """
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            self.tree[i-1] %= MOD
            i += i & -i

    def query(self, l, r):
        """ 区間和の取得 [l, r) """
        return (self.sum(r-1) - self.sum(l-1)) % MOD
    
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


class BIT:
    """ BIT汎用版 """

    def __init__(self, n, func, init):
        # 0-indexed
        n += 1
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.func = func
        self.init = init
        self.tree = [init] * nv
    
    def query(self, i):
        """ [0, i]の値を取得 """
        s = self.init
        i += 1
        while i > 0:
            s = self.func(s, self.tree[i-1])
            i -= i & -i
        return s
    
    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        i += 1
        while i <= self.size:
            self.tree[i-1] = self.func(self.tree[i-1], x)
            i += i & -i


class SegTree:
    """
    セグメント木
    1.update: i番目の値をxに更新する
    2.query: 区間[l, r)の値を得る
    """

    def __init__(self, n, func, intv, A=[]):
        """
        :param n: 要素数(0-indexed)
        :param func: 値の操作に使う関数(min, max, add, gcdなど)
        :param intv: 要素の初期値(単位元)
        :param A: 初期化に使うリスト(オプション)
        """
        self.n = n
        self.func = func
        self.intv = intv
        # nより大きい2の冪数
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.intv] * (n2 << 1)
        # 初期化の値が決まっている場合
        if A:
            # 1段目(最下段)の初期化
            for i in range(n):
                self.tree[n2+i] = A[i]
            # 2段目以降の初期化
            for i in range(n2-1, -1, -1):
                self.tree[i] = self.func(self.tree[i*2], self.tree[i*2+1])

    def update(self, i, x):
        """
        i番目の値をxに更新
        :param i: index(0-indexed)
        :param x: update value
        """
        i += self.n2
        self.tree[i] = x
        while i > 0:
            i >>= 1
            self.tree[i] = self.func(self.tree[i*2], self.tree[i*2+1])

    def add(self, i, x):
        self.update(i, self.get(i) + x)

    def query(self, a, b):
        """
        [a, b)の値を得る
        :param a: index(0-indexed)
        :param b: index(0-indexed)
        """
        l = a + self.n2
        r = b + self.n2
        s = self.intv
        while l < r:
            if r & 1:
                r -= 1
                s = self.func(s, self.tree[r])
            if l & 1:
                s = self.func(s, self.tree[l])
                l += 1
            l >>= 1
            r >>= 1
        return s

    def get(self, i):
        """ 一点取得 """
        return self.tree[i+self.n2]

    def all(self):
        """ 全区間[0, n)の取得 """
        return self.tree[1]

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

    def bisearch_fore(self, l, r, x, func):
        """ 区間[l,r]で左から最初にxに対して比較の条件を満たすような値が出現する位置 """

        ok = r + 1
        ng = l - 1
        while ng+1 < ok:
            mid = (ok+ng) // 2
            if func(self.query(l, mid+1), x):
                ok = mid
            else:
                ng = mid
        if ok != r + 1:
            return ok
        else:
            return -1

    def bisearch_back(self, l, r, x, func):
        """ 区間[l,r]で右から最初にxに対して比較の条件を満たすような値が出現する位置 """

        ok = l - 1
        ng = r + 1
        while ok+1 < ng:
            mid = (ok+ng) // 2
            if func(self.query(mid, r+1), x):
                ok = mid
            else:
                ng = mid
        if ok != l - 1:
            return ok
        else:
            return -1


class SegTreeIndex:
    """
    セグメント木(index取得対応版)
    1.update: i番目の値をxに更新する
    2.query: 区間[l, r)の値とindex(同値があった場合は一番左)を得る
    """

    def __init__(self, n, func, intv, A=[]):
        """
        :param n: 要素数(0-indexed)
        :param func: 値の操作に使う関数(min, max)
        :param intv: 要素の初期値(単位元)
        :param A: 初期化に使うリスト(オプション)
        """
        self.n = n
        self.func = func
        self.intv = (intv, -1)
        # nより大きい2の冪数
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.intv] * (n2 << 1)
        # 初期化の値が決まっている場合
        if A:
            # 1段目(最下段)の初期化
            for i in range(n):
                self.tree[n2+i] = (A[i], i)
                # 2段目以降の初期化
            for i in range(n2-1, -1, -1):
                self.tree[i] = self.compare(self.tree[i*2], self.tree[i*2+1])

    def compare(self, a, b):
        if a[0] == b[0]:
            # 同値はindexが小さい方優先
            if a[1] <= b[1]:
                return a
            else:
                return b
        elif self.func(a[0], b[0]) == a[0]:
            return a
        else:
            return b

    def update(self, i, x):
        """
        i番目の値をxに更新
        :param i: index(0-indexed)
        :param x: update value
        """
        i += self.n2
        self.tree[i] = (x, i-self.n2)
        while i > 0:
            i >>= 1
            self.tree[i] = self.compare(self.tree[i*2], self.tree[i*2+1])

    def add(self, i, x):
        self.update(i, self.get(i) + x)

    def query(self, a, b):
        """
        [a, b)の値を得る
        :param a: index(0-indexed)
        :param b: index(0-indexed)
        """
        l = a + self.n2
        r = b + self.n2
        s = self.intv
        while l < r:
            if r & 1:
                r -= 1
                s = self.compare(s, self.tree[r])
            if l & 1:
                s = self.compare(s, self.tree[l])
                l += 1
            l >>= 1
            r >>= 1
        return s

    def get(self, i):
        """ 一点取得 """
        return self.tree[i+self.n2][0]

    def all(self):
        """ 全区間[0, n)の取得 """
        return self.tree[1]

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()


class StarrySkyTree:
    """ Starry Sky Tree(区間加算・区間最小(大)値取得) """

    def __init__(self, N, func, intv):
        self.intv = intv
        self.func = func
        LV = (N-1).bit_length()
        self.N0 = 2**LV
        self.data = [0]*(2*self.N0)
        self.lazy = [0]*(2*self.N0)

    # 伝搬される区間のインデックス(1-indexed)を全て列挙するgenerator
    def gindex(self, l, r):
        L = l + self.N0; R = r + self.N0
        lm = (L // (L & -L)) >> 1
        rm = (R // (R & -R)) >> 1
        while L < R:
            if R <= rm:
                yield R
            if L <= lm:
                yield L
            L >>= 1; R >>= 1
        while L:
            yield L
            L >>= 1

    # 遅延評価の伝搬処理
    def propagates(self, *ids):
        # 1-indexedで単調増加のインデックスリスト
        for i in reversed(ids):
            v = self.lazy[i-1]
            if not v:
                continue
            self.lazy[2*i-1] += v; self.lazy[2*i] += v
            self.data[2*i-1] += v; self.data[2*i] += v
            self.lazy[i-1] = 0

    def update(self, l, r, x):
        """ 区間[l,r)の値にxを加算 """

        # 1. lazyの値は伝搬させない
        # 2. 区間[l,r)のdata, lazyの値を更新
        L = self.N0 + l; R = self.N0 + r
        while L < R:
            if R & 1:
                R -= 1
                self.lazy[R-1] += x; self.data[R-1] += x
            if L & 1:
                self.lazy[L-1] += x; self.data[L-1] += x
                L += 1
            L >>= 1; R >>= 1
        # 3. 更新される区間を部分的に含んだ区間のdataの値を更新 (lazyの値を考慮)
        for i in self.gindex(l, r):
            self.data[i-1] = self.func(self.data[2*i-1], self.data[2*i]) + self.lazy[i-1]

    def query(self, l, r):
        """ 区間[l,r)の最小(大)値を取得 """

        # 1. トップダウンにlazyの値を伝搬
        self.propagates(*self.gindex(l, r))
        L = self.N0 + l; R = self.N0 + r

        # 2. 区間[l, r)の最小(大)値を求める
        s = self.intv
        while L < R:
            if R & 1:
                R -= 1
                s = self.func(s, self.data[R-1])
            if L & 1:
                s = self.func(s, self.data[L-1])
                L += 1
            L >>= 1; R >>= 1
        return s

    def print(self, N):
        for i in range(N):
            print(self.query(i, i+1), end=' ')
        print()


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

    def get(self, i):
        return self.query(i, i+1)
    
    def update(self, i, x):
        self.add(i, i+1, x - self.get(i))

class SparseTable:

    def __init__(self, A, func):
        self.N = len(A)
        self.func = func
        h = 0
        while 1<<h <= self.N:
            h += 1
        self.dat = list2d(h, 1<<h, 0)
        self.height = [0] * (self.N+1)

        for i in range(2, self.N+1):
            self.height[i] = self.height[i>>1] + 1
        for i in range(self.N):
            self.dat[0][i] = A[i]
        for i in range(1, h):
            for j in range(self.N):
                self.dat[i][j] = self.func(self.dat[i-1][j], self.dat[i-1][min(j+(1<<(i-1)), self.N-1)])
        
    def get(self, l, r):
        """ 区間[l,r)でのmin,maxを取得 """

        if l >= r or r > self.N:
            raise Exception
        a = self.height[r-l]
        return self.func(self.dat[a][l], self.dat[a][r-(1<<a)])

    def bisearch_fore(self, l, r, x, func):
        """ 区間[l,r]で左から最初にxに対して比較の条件を満たすような値が出現する位置 """

        ok = r + 1
        ng = l - 1
        while ng+1 < ok:
            mid = (ok+ng) // 2
            if func(self.get(l, mid+1), x):
                ok = mid
            else:
                ng = mid
        if ok != r + 1:
            return ok
        else:
            return -1

    def bisearch_back(self, l, r, x, func):
        """ 区間[l,r]で右から最初にxに対して比較の条件を満たすような値が出現する位置 """

        ok = l - 1
        ng = r + 1
        while ok+1 < ng:
            mid = (ok+ng) // 2
            if func(self.get(mid, r+1), x):
                ok = mid
            else:
                ng = mid
        if ok != l - 1:
            return ok
        else:
            return -1


def slide_min(func, A, k):
    """ スライド最小値(演算、元の数列A、欲しい区間長k) """
    from collections import deque

    que = deque()
    res = []
    N = len(A)
    for i in range(N):
        while que and func(A[que[-1]], A[i]) == A[i]:
            que.pop()
        que.append(i)
        if i-k+1 >= 0:
            res.append(A[que[0]])
            if que[0] == i-k+1:
                que.popleft()
    return res
