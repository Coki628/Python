"""
・自力AC！
・貰うDP高速化、BIT
・いわゆる区間和取得でDP高速化するやつ。ちゃんと整えれば累積和でもいけると思ったけど、
　多分logついても間に合うかな、と判断してBITで雑に区間和取った。
・最初WAでなんでかなーって思ったけど、MOD取る時に右端-左端のとこで取れてなかった。。
　ともあれBITがMOD数え上げでも問題なく機能することは確認できた。
・計算量20万のN*10*logNだから結構あると思ったけどpypyAC0.5秒。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
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

N, K = MAP()
LR = []
for i in range(K):
    l, r = MAP()
    LR.append((l, r))

dp = BIT(N+1)
dp.add(1, 1)
for i in range(2, N+1):
    for l, r in LR:
        cnt = dp.query(max(i-r, 0), max(i-l+1, 0))
        dp.add(i, cnt)
ans = dp.get(N)
print(ans)
