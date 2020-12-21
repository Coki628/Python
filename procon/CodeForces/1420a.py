"""
・自力AC！
・座標圧縮、転倒数、BIT
・BITでの転倒数数え上げをやるけど、同値を含む数列なので、自分より1つ手前まででやる。
・愚直にちゃんと数えたけど、これよく見たら転倒数が最大(N*(N-1)//2)の時以外全部YESでいいんじゃんね。。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class BIT:

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
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s

    def add(self, i, x):
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

    def query(self, l, r):
        return self.sum(r-1) - self.sum(l-1)
    
    def get(self, i):
        return self.query(i, i+1)

    def update(self, i, x):
        self.add(i, x - self.get(i))

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

def compress(S):

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

for _ in range(INT()):
    N = INT()
    A = LIST()

    S = set(A+[0])
    zipped, _ = compress(S)
    for i in range(N):
        A[i] = zipped[A[i]]

    bit = BIT(N+1)
    ans = 0
    limit = N*(N-1) // 2 - 1
    for i in range(N-1, -1, -1):
        ans += bit.sum(A[i]-1)
        bit.add(A[i], 1)

    if ans <= limit:
        YES()
    else:
        NO()
