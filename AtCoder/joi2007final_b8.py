"""
・ローリングハッシュ、LCP
・ロリハlcp使ってみた。MLEだしTLE。。昔のJOI色々制約きついよね。
・ていうか、言語アップデートで新しくなったpypyだとデフォで64MB以上使ってて
　どうやっても無理ゲーって感じっぽい。。
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

class RollingHash:

    """ 文字列stringの部分文字列のハッシュを構築する：O(N) """
    def __init__(self, string):
        self.n = len(string)
        self.BASE = 1234
        self.MASK30 = (1 << 30) - 1
        self.MASK31 = (1 << 31) - 1
        self.MASK61 = (1 << 61) - 1
        self.MOD = self.MASK61
        self.hash = [0] * (self.n + 1)
        self.pow = [1] * (self.n + 1)
 
        for i, char in enumerate(string):
            self.hash[i + 1] = self.calc_mod(self.mul(self.hash[i], self.BASE) + ord(char))
            self.pow[i + 1] = self.calc_mod(self.mul(self.pow[i], self.BASE))
 
    def calc_mod(self, x):
        """ x mod 2^61-1 を返す """
        xu = x >> 61
        xd = x & self.MASK61
        x = xu + xd
        if x >= self.MOD:
            x -= self.MASK61
        return x
 
    def mul(self, a, b):
        """ a*b mod 2^61-1 を返す """
        au = a >> 31
        ad = a & self.MASK31
        bu = b >> 31
        bd = b & self.MASK31
        mid = ad * bu + au * bd
        midu = mid >> 30
        midd = mid & self.MASK30
        return self.calc_mod(au * bu * 2 + midu + (midd << 31) + ad * bd)
 
    def get_hash(self, l, r):
        """ string[l,r)のハッシュ値を返す：O(1) """
        res = self.calc_mod(self.hash[r] - self.mul(self.hash[l], self.pow[r - l]))
        return res
 
    def merge(self, h1, h2, length2):
        """ ハッシュ値h1と長さlength2のハッシュ値h2を結合する：O(1) """
        return self.calc_mod(self.mul(h1, self.pow[length2]) + h2)
 
    def get_lcp(self, l1, r1, l2, r2):
        """
            string[l1:r2]とstring[l2:r2]の最長共通接頭辞(Longest Common Prefix)の
            長さを求める：O(log|string|)
        """
        ng = min(r1 - l1, r2 - l2) + 1
        ok = 0
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            if self.get_hash(l1, l1 + mid) == self.get_hash(l2, l2 + mid):
                ok = mid
            else:
                ng = mid
        return ok

S = input()
N = len(S)
T = input()
M = len(T)
NM = N + M

rh = RollingHash(S+T)
ans = 0
for i in range(N):
    for j in range(N, NM):
        ans = max(ans, rh.get_lcp(i, N, j, NM))
print(ans)
