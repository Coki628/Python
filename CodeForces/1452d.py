"""
・なんとか自力AC！
・1次元DP、高速化
・確率を聞かれてるけど、本質は数え上げDP。条件を満たすように電波塔を設置するには、
　それぞれに奇数長の区間をカバーさせればいいので、自分の1,3,5...個前からは遷移してこれる。
・最初、1個飛ばし累積和をやろうとしたらバグりまくって終了。。
　log付くけどBIT2本に偶数位置と奇数位置を管理させて区間和取得、でやったらすぐうまくいった。。
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
INF = 10**19
MOD = 998244353
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

def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

N = INT()

bit1 = BIT(N+1)
bit2 = BIT(N+1)
bit1.add(0, 1)
for i in range(1, N+1):
    if i % 2 == 0:
        bit1.add(i, bit2.query(0, i))
    else:
        bit2.add(i, bit1.query(0, i))
if N % 2 == 0:
    cnt = bit1.get(N)
else:
    cnt = bit2.get(N)

ans = fermat(cnt, pow(2, N, MOD), MOD)
print(ans)
