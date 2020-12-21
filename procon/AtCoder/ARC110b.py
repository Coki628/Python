"""
・なんとか自力AC。きっつかった。。
・(連続した)部分文字列、ローリングハッシュ、周期
・とりあえず110110***と文字列Tが同時に終わるところを1周期として、それを3*10^10まで進める方針にする。
・最初の1周と、最後の余りだけ愚直に数えればいいんだけど、20万の文字列で文字列比較毎回やるのは無理なので、
　ロリハを使う。
・他、細かい微調整が色々必要だったんだけど、何とか時間内にAC。。
　詳細はソース内コメント。
"""

import sys
from math import gcd

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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def lcm(x, y): return (x * y) // gcd(x, y)

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

N = INT()
T = input()

K = 3 * 10**10
M = 3
S = '110'
L = lcm(N, M)

# 長さLを1周期とする
S2 = []
for i in range(L):
    S2.append(S[i%M])

rh1 = RollingHash(S2*2)
rh2 = RollingHash(T)
thash = rh2.get_hash(0, N)
cnt = 0
# add[i] := 区間からi文字はみ出たところで得られるカウント
add = [0] * L
# sub := はみ出たところでカウントした総数
sub = 0
# 1周期分の愚直シミュレーション
for l in range(L):
    r = l + N
    if rh1.get_hash(l, r) == thash:
        cnt += 1
        if r > L:
            sub += 1
            add[r%L] = 1
for i in range(L-1):
    add[i+1] += add[i]

d, m = divmod(K, L)
# d周期分まとめて足す
ans = cnt * d
# 最後に区間外にはみ出たところを引く
ans -= sub
# 今回の余りでカバーできる分は戻す
ans += add[m]
# 開始位置からして今回の余りに属する分を足す
for l in range(m-N+1):
    r = l + N
    if rh1.get_hash(l, r) == thash:
        ans += 1
print(ans)
