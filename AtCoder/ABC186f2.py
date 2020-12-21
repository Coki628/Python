"""
・後から色々考えて、やっぱり余事象の方でもできるのでは、
　って思って自分が作ってWAだった方を元に修正、無事こっちでもAC。
・ざっくり2点、1行目だけ障害物の後ろを全部埋めるってことと、
　1列目で障害物が出たらそれ以降は区間和の左端を先頭からにする、ってのを直したら通った。
・まあでもこっちのがちょっとややこしい気がするし、ソートして障害物のある行だけ見てるのも、
　行数が10^9ならともかく、今回は10^5なんだから全部見るようにした方が全然考えやすい。
"""

import sys
from operator import itemgetter

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
for i in range(M):
    h, w = MAP()
    HW.append((h, w))

HW.sort(key=itemgetter(1), reverse=1)
HW.sort(key=itemgetter(0))
HW.append((H+1, W+1))

cnt = 0
bit = BIT(W+1)
col1ng = 0
for i in range(M):
    if HW[i][1] == 1:
        col1ng = 1
    if HW[i][0] == 1:
        # 1行目だけ、障害物の後ろを全部埋める
        j = HW[i][1]
        while j <= W and bit.get(j) == 0:
            bit.add(j, 1)
            j += 1
    else:
        if bit.get(HW[i][1]) == 0:
            bit.add(HW[i][1], 1)
    if HW[i][0] != HW[i+1][0]:
        # 1列目が既に下りてこれない行かどうかで場合分け
        if col1ng:
            rowcnt = bit.query(0, W+1)
            hlen = HW[i+1][0] - HW[i][0]
            cnt += rowcnt*hlen
        else:
            cnt += bit.query(HW[i][1], W+1)
ans = H*W - cnt
print(ans)
