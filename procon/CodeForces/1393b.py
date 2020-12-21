"""
・なんとか自力AC
・indexセグ木
・最初ちゃんと問題読めてなくて、正方形か長方形で何でもいいから2つだと思ってたけど、
　よく読んだら1つは正方形確定で2つ目がどっちでもいい、だった。。
・それに合わせて条件整えたらAC。。
・と思ったらシステス落ちてた。。TLE。。。C++でindexセグ木整えて出したらAC。
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

class SegTreeIndex:

    def __init__(self, n, func, intv, A=[]):
        self.n = n
        self.func = func
        self.intv = (intv, -1)
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.intv] * (n2 << 1)
        if A:
            for i in range(n):
                self.tree[n2+i] = (A[i], i)
            for i in range(n2-1, -1, -1):
                self.tree[i] = self.compare(self.tree[i*2], self.tree[i*2+1])

    def compare(self, a, b):
        if a[0] == b[0]:
            if a[1] <= b[1]:
                return a
            else:
                return b
        elif self.func(a[0], b[0]) == a[0]:
            return a
        else:
            return b

    def update(self, i, x):
        i += self.n2
        self.tree[i] = (x, i-self.n2)
        while i > 0:
            i >>= 1
            self.tree[i] = self.compare(self.tree[i*2], self.tree[i*2+1])

    def add(self, i, x):
        self.update(i, self.get(i) + x)

    def query(self, a, b):
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
        return self.tree[i+self.n2][0]

    def all(self):
        return self.tree[1]

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

N = INT()
A = LIST()
MX = 10 ** 5
C = [0] * (MX+1)
for a in A:
    C[a] += 1

seg = SegTreeIndex(MX+1, max, -INF, C)

def check():
    mx, idx = seg.all()
    tmp = []
    if mx >= 4:
        seg.update(idx, mx-4)
        tmp.append((idx, 4))
        mx, idx = seg.all()
        if mx >= 4:
            for idx, val in tmp:
                seg.add(idx, val)
            # 4, 4
            return True
        elif mx >= 2:
            seg.update(idx, mx-2)
            tmp.append((idx, 2))
            mx, idx = seg.all()
            if mx >= 2:
                for idx, val in tmp:
                    seg.add(idx, val)
                # 4, 2, 2
                return True
            else:
                for idx, val in tmp:
                    seg.add(idx, val)
                # 4, 2, x
                return False
        else:
            for idx, val in tmp:
                seg.add(idx, val)
            # 4, x
            return False
    else:
        # x
        return False

for _ in range(INT()):
    op, x = input().split()
    x = int(x)

    if op == '+':
        seg.add(x, 1)
    else:
        seg.add(x, -1)
    
    if check():
        YES()
    else:
        NO()
