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
INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

class SegTree:

    def __init__(self, n, func, intv, A=[]):
        self.n = n
        self.func = func
        self.intv = intv
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.intv] * (n2 << 1)
        if A:
            for i in range(n):
                self.tree[n2+i] = A[i]
            for i in range(n2-1, -1, -1):
                self.tree[i] = self.func(self.tree[i*2], self.tree[i*2+1])

    def update(self, i, x):
        i += self.n2
        self.tree[i] = x
        while i > 0:
            i >>= 1
            self.tree[i] = self.func(self.tree[i*2], self.tree[i*2+1])

    def add(self, i, x):
        self.update(i, self.get(i) + x)

    def query(self, a, b):
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
        return self.tree[i+self.n2]

    def all(self):
        return self.tree[1]

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

def bisearch_max(mn, mx, func):
    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

N, M = MAP()
AB = []
for i in range(N):
    a, b = MAP()
    AB.append((a, b))
CD = []
for i in range(M):
    c, d = MAP()
    CD.append((c, d))

AB.sort()
C, D = zip(*CD)
L = max(C)
seg = SegTree(L+2, max, -INF)
for c, d in CD:
    seg.update(c, d)

def check(m):
    res = 0
    for a, b in AB:
        if a+m >= L+1:
            break
        res = max(res, max(seg.query(min(a+m, L+1), L+2)-b+1, 0))
    return res == y

ans = INF
x = 0
while x < L+2:
    y = 0
    for a, b in AB:
        if a+x >= L+1:
            break
        y = max(y, max(seg.query(min(a+x, L+1), L+2)-b+1, 0))
    ans = min(ans, x+y)
    x = bisearch_max(x, L+2, check) + 1
print(ans)
