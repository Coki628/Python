# -*- coding: utf-8 -*-

# 各種インポート
import sys, re
from collections import deque, defaultdict, Counter
from math import (
    sqrt, hypot, factorial, log10, log2,
    pi, sin, cos, acos, atan2, radians, degrees,
)
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
from operator import itemgetter, mul, add, xor
# from copy import copy, deepcopy
from functools import reduce, partial, lru_cache
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits
from os.path import commonprefix

# def input(): return sys.stdin.buffer.readline().strip()
def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(li): return reduce(lcm, li, 1)
def gcd_list(li): return reduce(gcd, li, 0)
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

# 標準ライブラリのよりこっちのが速い(ただし2次元限定)
def deepcopy(li): return [x[:] for x in li]

# numpy系
import numpy as np
from scipy.sparse.csgraph import dijkstra, floyd_warshall

# 調査用
# import matplotlib.pyplot as plt 
# import pandas as pd

# 再帰呼び出しの回数制限(デフォルト1000)
sys.setrecursionlimit(10 ** 9)
# 再帰関数の前にこれ書くと速くなったりするらしい。
@lru_cache(maxsize=None)
def rec():
    rec()

# 小数点以下10桁まで表示(これやんないと自動でeとか使われる)
'{:.10f}'.format(3.1415)

# 文字列リバース
s = ''
s = s[::-1]

# 行列入れ替え
# li2 = list(zip(*li1))

# int -> bin(str)
num = 1234
num = format(num, 'b')
# int -> bin(str) (8桁0埋め)
num = format(aN[i], '08b')
# bin(str) => int
num = int(num, 2)
# int -> 10進4桁0埋め(str)
num = format(num, '04')

# 二番目の要素でソート
aN = [[1, 2], [3, 1]]
aN.sort(key=lambda x: x[1])
# こっちのがちょっと速い
aN.sort(key=itemgetter(1))

# 四捨五入で整数に丸める
def round(x): return int((x*2+1) // 2)

# modの除算(フェルマーの小定理)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

# 配列要素全部掛け(総乗)
prod = partial(reduce, mul)
# これでもよさげ
def prod(nums): return reduce(mul, nums, 1)
prod([1, 2, 3])
np.prod([1, 2, 3])

# 余りの切り上げ(3つとも同じ)
# def ceil(a, b):
#     (a + b - 1) // b
#     (a - 1) // b + 1
#     return -(-a // b)

# 最小公倍数
def lcm(x, y): return (x * y) // gcd(x, y)
# reduce(使う関数, 足し合わせるリスト, 初期値)
def lcm_list(nums): return reduce(lcm, nums, initial=1)

# 1からnまでの等差数列の和
def get_sum(n): return (1+n)*n//2

def get_sum(a, l, n):
    """ 等差数列の和：(初項a, 末項l, 項数n) """
    return (a+l) * n // 2

def get_sum(a, d, n):
    """ 等差数列の和：(初項a, 公差d, 項数n) """
    return (2*a + (n-1)*d) * n // 2

def digit_sum(n):
    """ 桁和：O(logN) """

    ans = 0
    while n > 0:
        ans += n % 10
        n //= 10
    return ans

def is_prime(num):
    """ 素数判定 """
    from math import sqrt

    if num < 2: 
        return False
    if num in [2, 3, 5]: 
        return True
    if num % 2 == 0 or num % 3 == 0 or num % 5 == 0:
        return False
    # 疑似素数(2でも3でも割り切れない数字)で次々に割っていく
    prime = 7
    step = 4
    num_sqrt = sqrt(num)
    while prime <= num_sqrt:
        if num % prime == 0:
            return False
        prime += step
        step = 6 - step
    return True

def eratosthenes_sieve(n):
    """ 素数列挙(エラトステネスの篩) """

    table = [0] * (n + 1)
    prime_list = []
    for i in range(2, n + 1):
        if table[i] == 0:
            prime_list.append(i)
            for j in range(i + i, n + 1, i):
                table[j] = 1
    return prime_list

def factorize(num: int) -> dict:
    """ 素因数分解 """
    from math import sqrt
    from collections import Counter

    d = Counter()
    # 終点はルート切り捨て+1
    for i in range(2, int(sqrt(num))+1):
        # 素因数分解：小さい方から割れるだけ割って素数をカウント
        while num % i == 0:
            num //= i
            d[i] += 1
        # 1まで分解したら終了
        if num == 1:
            break
    # 最後に残ったnumは素数(ただし1^1は1^0なので数に入れない)
    if num != 1:
        d[num] += 1
    return d

def divisors(N: int) -> set:
    """ 約数の列挙・個数 """
    from math import sqrt

    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て
    for i in range(2, int(sqrt(N))+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

def LIS(A: list, equal=False) -> list:
    """ 最長増加部分列 """
    from operator import gt, ge
    from bisect import bisect_left, bisect_right

    # デフォルトは狭義のLIS(同値を含まない)
    compare = gt if not equal else ge
    bisect = bisect_left if not equal else bisect_right
    L = [A[0]]
    for a in A[1:]:
        if compare(a, L[-1]):
            # Lの末尾よりaが大きければ増加部分列を延長できる
            L.append(a)
        else:
            # そうでなければ、「aより小さい最大要素の次」をaにする
            # 該当位置は、二分探索で特定できる
            L[bisect(L, a)] = a
    return L

def fft(A, B, L):
    """ 
    高速フーリエ変換(FFT)
        A：出現回数をカウントしたリスト
        B：出現回数をカウントしたリスト
        L：戻り値配列の長さ(max(A)+max(B)より大きい2冪を渡す)
    """
    import numpy as np
    from numpy.fft import rfft, irfft

    # FFT
    res = irfft(rfft(A, L) * rfft(B, L), L)
    # 四捨五入して整数に
    res = np.rint(res).astype(np.int64)
    return res

def fft(A, B):
    """ 
    高速フーリエ変換(FFT)
    """
    import numpy as np
    from numpy.fft import rfft, irfft

    # 出現数カウント
    MAXA = max(A)
    MAXB = max(B)
    C1 = [0] * (MAXA+1)
    C2 = [0] * (MAXB+1)
    for a in A:
        C1[a] += 1
    for b in B:
        C2[b] += 1
    # max(A)+max(B)より大きい2冪
    L = 1
    k = 0
    while L <= MAXA + MAXB:
        k += 1
        L = 2**k
    # FFT
    res = irfft(rfft(C1, L) * rfft(C2, L), L)
    # 四捨五入して整数に
    res = np.rint(res).astype(np.int64)
    return res


class ModTools:
    """ 階乗たくさん使う時用のテーブル準備 """

    def __init__(self, MAX, MOD):
        """ MAX：階乗に使う数値の最大以上まで作る """
        
        MAX += 1
        self.MAX = MAX
        self.MOD = MOD
        # 階乗テーブル
        factorial = [1] * MAX
        factorial[0] = factorial[1] = 1
        for i in range(2, MAX):
            factorial[i] = factorial[i-1] * i % MOD
        # 階乗の逆元テーブル
        inverse = [1] * MAX
        # powに第三引数入れると冪乗のmod付計算を高速にやってくれる
        inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
        for i in range(MAX-2, -1, -1):
            # 最後から戻っていくこのループならMAX回powするより処理が速い
            inverse[i] = inverse[i+1] * (i+1) % MOD
        self.fact = factorial
        self.inv = inverse
    
    def nCr(self, n, r):
        """ 組み合わせの数 (必要な階乗と逆元のテーブルを事前に作っておく) """

        if n < r: return 0
        # 10C7 = 10C3
        r = min(r, n-r)
        # 分子の計算
        numerator = self.fact[n]
        # 分母の計算
        denominator = self.inv[r] * self.inv[n-r] % self.MOD
        return numerator * denominator % self.MOD

    def nHr(self, n, r):
        """ 重複組み合わせ """

        # r個選ぶところにN-1個の仕切りを入れる
        return self.nCr(r+n-1, r)

    def nPr(self, n, r):
        """ 順列 """

        if n < r: return 0
        return self.fact[n] * self.inv[n-r] % self.MOD

    def div(self, x, y):
        """ MOD除算 """

        return x * pow(y, self.MOD-2, self.MOD) % self.MOD


def nCr(n, r, MOD):
    """ 組み合わせの数(大きいnに対して使用する。計算量はr) """

    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    num = den = 1
    # 分子の計算
    for i in range(n, n-r, -1):
        num *= i % MOD
        num %= MOD
    # 分母の計算
    for i in range(r, 0, -1):
        den *= i % MOD
        den %= MOD
    # logがつくため、MOD除算は最後の1回だけにする
    return num * pow(den, MOD-2, MOD) % MOD

def init_fact_inv(MAX: int, MOD: int):
    """ 階乗たくさん使う時用のテーブル準備

    Parameters
    ----------
        MAX：階乗に使う数値の最大以上まで作る
        MOD
    Returns
    -------
        factorial (list<int>), inverse (list<int>)
    """

    MAX += 1
    # 階乗テーブル
    factorial = [1] * MAX
    factorial[0] = factorial[1] = 1
    for i in range(2, MAX):
        factorial[i] = factorial[i-1] * i % MOD
    # 階乗の逆元テーブル
    inverse = [1] * MAX
    # powに第三引数入れると冪乗のmod付計算を高速にやってくれる
    inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
    for i in range(MAX-2, 0, -1):
        # 最後から戻っていくこのループならMAX回powするより処理が速い
        inverse[i] = inverse[i+1] * (i+1) % MOD
    return factorial, inverse

def nCr(n, r, factorial, inverse):
    """ 組み合わせの数 (必要な階乗と逆元のテーブルを事前に作っておく) """

    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = inverse[r] * inverse[n-r] % MOD
    return numerator * denominator % MOD

def init_factorial(MAX: int) -> list:
    """ テーブル準備MODなし版 """

    MAX += 1
    # 階乗テーブル
    factorial = [1] * MAX
    factorial[0] = factorial[1] = 1
    for i in range(2, MAX):
        factorial[i] = factorial[i-1] * i
    return factorial

def nCr(n, r):
    """ 組み合わせの数(必要な階乗のテーブルを事前に作っておく) """

    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = factorial[r] * factorial[n-r]
    return numerator // denominator

def init_fact_log(MAX: int) -> list:
    """ テーブル準備logでやる版 """
    from math import log10

    MAX += 1
    fact_log = [0] * MAX
    for i in range(1, MAX):
        fact_log[i] = fact_log[i-1] + log10(i)
    return fact_log

def nCr(n, r, fact_log):
    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    return round(pow(10, fact_log[n] - fact_log[r] - fact_log[n-r]))

def nCr(n, r):
    """ 事前テーブルなし組み合わせ簡易版 """
    from math import factorial

    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    return factorial(n) // (factorial(r) * factorial(n-r))

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索 """

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索 """

    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

def bisearch_min(mn, mx, func, EPS):
    """ 条件を満たす最小値を見つける二分探索(小数用) """

    ok = mx
    ng = mn
    while ng+EPS < ok:
        mid = (ok+ng) / 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

def bisearch_max(mn, mx, func, EPS):
    """ 条件を満たす最大値を見つける二分探索(小数用) """

    ok = mn
    ng = mx
    while ok+EPS < ng:
        mid = (ok+ng) / 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

def compress(S):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

def shakutori(N, K, A):
    """ 尺取法 """

    l = r = ans = 0
    sm = 0
    while l < N:
        # ここのand以下の条件は問題によって変わる
        while r < N and sm + A[r] <= K:
            sm += A[r]
            r += 1
        # ここで求める答えに足したりmax取ったりする
        ans += r - l
        # 左が右に追いついたら、右も左に合わせて+1する
        if l == r:
            # 左右同時に動くので、何も引く必要はない
            r += 1
        else:
            # 左を動かす分、合計から引く
            sm -= A[l]
        l += 1

# 最大32ビット
def popcount(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

# 最大128ビット
def popcount(n):
    c = (n & 0x55555555555555555555555555555555) + ((n>>1) & 0x55555555555555555555555555555555)
    c = (c & 0x33333333333333333333333333333333) + ((c>>2) & 0x33333333333333333333333333333333)
    c = (c & 0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f) + ((c>>4) & 0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f)
    c = (c & 0x00ff00ff00ff00ff00ff00ff00ff00ff) + ((c>>8) & 0x00ff00ff00ff00ff00ff00ff00ff00ff)
    c = (c & 0x0000ffff0000ffff0000ffff0000ffff) + ((c>>16) & 0x0000ffff0000ffff0000ffff0000ffff)
    c = (c & 0x00000000ffffffff00000000ffffffff) + ((c>>32) & 0x00000000ffffffff00000000ffffffff)
    c = (c & 0x0000000000000000ffffffffffffffff) + ((c>>64) & 0x0000000000000000ffffffffffffffff)
    return c

# 最大65536ビット
NN = 16
KK = (1 << (1 << NN)) - 1
I0 = KK // 3
I1 = KK // 5
I2 = KK // 17
I3 = KK // 257
def popcount(x):
    x -= (x >> 1) & I0
    x = (x & I1) + ((x >> 2) & I1)
    x = (x + (x >> 4)) & I2
    x = (x + (x >> 8)) & I3
    for k in range(4, 16):
        x += x >> 2**k
    return x & 0xffff

# 最大262144ビット
NN = 18
KK = (1 << (1 << NN)) - 1
I0 = KK // 3
I1 = KK // 5
I2 = KK // 17
I3 = KK // 257
I4 = KK // 65537
def popcount(x):
    x -= (x >> 1) & I0
    x = (x & I1) + ((x >> 2) & I1)
    x = (x + (x >> 4)) & I2
    x = (x + (x >> 8)) & I3
    x = (x + (x >> 16)) & I4
    for k in range(5, 18):
        x += x >> 2**k
    return x & 0x3ffff

# 無制限
def popcount(i):
    return bin(i).count('1')


class Geometry:
    """ 幾何学計算用クラス """

    def __init__(self, EPS):
        self.EPS = EPS

    def add(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return (x1+x2, y1+y2)

    def sub(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return (x1-x2, y1-y2)

    def mul(self, a, b):
        x1, y1 = a
        if not isinstance(b, tuple):
            return (x1*b, y1*b)
        x2, y2 = b 
        return (x1*x2, y1*y2)

    def div(self, a, b):
        x1, y1 = a
        if not isinstance(b, tuple):
            return (x1/b, y1/b)
        x2, y2 = b
        return (x1/x2, y1/y2)

    def abs(self, a):
        from math import hypot
        x1, y1 = a
        return hypot(x1, y1)

    def norm(self, a):
        x, y = a
        return x**2 + y**2

    def dot(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return x1*x2 + y1*y2

    def cross(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return x1*y2 - y1*x2

    def project(self, seg, p):
        """ 線分segに対する点pの射影 """

        p1, p2 = seg
        base = self.sub(p2, p1)
        r = self.dot(self.sub(p, p1), base) / self.norm(base)
        return self.add(p1, self.mul(base, r))

    def reflect(self, seg, p):
        """ 線分segを対称軸とした点pの線対称の点 """

        return self.add(p, self.mul(self.sub(self.project(seg, p), p), 2))

    def ccw(self, p0, p1, p2):
        """ 線分p0,p1から線分p0,p2への回転方向 """

        a = self.sub(p1, p0)
        b = self.sub(p2, p0)
        # 反時計回り
        if self.cross(a, b) > self.EPS: return 1
        # 時計回り
        if self.cross(a, b) < -self.EPS: return -1
        # 直線上(p2 => p0 => p1)
        if self.dot(a, b) < -self.EPS: return 2
        # 直線上(p0 => p1 => p2)
        if self.norm(a) < self.norm(b): return -2
        # 直線上(p0 => p2 => p1)
        return 0

    def intersect(self, seg1, seg2):
        """ 線分seg1と線分seg2の交差判定 """

        p1, p2 = seg1
        p3, p4 = seg2
        return (
            self.ccw(p1, p2, p3) * self.ccw(p1, p2, p4) <= 0
            and self.ccw(p3, p4, p1) * self.ccw(p3, p4, p2) <= 0
        )

    def get_distance_LP(self, line, p):
        """ 直線lineと点pの距離 """

        p1, p2 = line
        return abs(self.cross(self.sub(p2, p1), self.sub(p, p1)) / self.abs(self.sub(p2, p1)))

    def get_distance_SP(self, seg, p):
        """ 線分segと点pの距離 """

        p1, p2 = seg
        if self.dot(self.sub(p2, p1), self.sub(p, p1)) < 0: return self.abs(self.sub(p, p1))
        if self.dot(self.sub(p1, p2), self.sub(p, p2)) < 0: return self.abs(self.sub(p, p2))
        return self.get_distance_LP(seg, p)

    def get_distance_SS(self, seg1, seg2):
        """ 線分seg1と線分seg2の距離 """

        p1, p2 = seg1
        p3, p4 = seg2
        if self.intersect(seg1, seg2): return 0
        return min(
            self.get_distance_SP(seg1, p3), self.get_distance_SP(seg1, p4),
            self.get_distance_SP(seg2, p1), self.get_distance_SP(seg2, p2),
        )

    def get_cross_pointSS(self, seg1, seg2):
        """ 線分seg1と線分seg2の交点 """

        p1, p2 = seg1
        p3, p4 = seg2
        base = self.sub(p4, p3)
        dist1 = abs(self.cross(base, self.sub(p1, p3)))
        dist2 = abs(self.cross(base, self.sub(p2, p3)))
        t = dist1 / (dist1+dist2)
        return self.add(p1, self.mul(self.sub(p2, p1), t))
    
    def intersectCL(self, c, line):
        """ 円cと直線lineの交差判定 """

        x, y, r = c
        return self.get_distance_SP(line, (x, y)) <= r

    def get_cross_pointCL(self, c, line):
        """ 円cと直線lineの交点 """
        from math import sqrt

        if not self.intersectCL(c, line): return -1
        x, y, r = c
        p1, p2 = line
        pr = self.project(line, (x, y))
        e = self.div(self.sub(p2, p1), self.abs(self.sub(p2, p1)))
        base = sqrt(r*r - self.norm(self.sub(pr, (x, y))))
        return [self.add(pr, self.mul(e, base)), self.sub(pr, self.mul(e, base))]
    
    def arg(self, p):
        from math import atan2
        x, y = p
        return atan2(y, x)
    
    def polar(self, a, r):
        from math import sin, cos
        return (cos(r)*a, sin(r)*a)
    
    def intersectCC(self, c1, c2):
        """ 円c1と円c2の交差判定 """
        from math import hypot

        x1, y1, r1 = c1
        x2, y2, r2 = c2
        return hypot(x1-x2, y1-y2) <= r1 + r2
    
    def get_cross_pointCC(self, c1, c2):
        """ 円c1と円c2の交点 """
        from math import acos

        if not self.intersectCC(c1, c2): return -1
        x1, y1, r1 = c1
        x2, y2, r2 = c2
        try:
            d = self.abs(self.sub((x1, y1), (x2, y2)))
            a = acos((r1*r1+d*d-r2*r2) / (2*r1*d))
            t = self.arg(self.sub((x2, y2), (x1, y1)))
            return [self.add((x1, y1), self.polar(r1, t+a)), self.add((x1, y1), self.polar(r1, t-a))]
        except:
            # 一方が他方を内包しちゃってる場合等はここに飛ぶ(はず)
            return -1
