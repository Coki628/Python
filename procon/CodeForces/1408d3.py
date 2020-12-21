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
        if l >= r or r > self.N:
            raise Exception
        a = self.height[r-l]
        return self.func(self.dat[a][l], self.dat[a][r-(1<<a)])

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

C, D = zip(*CD)
L = max(C)
li = [-INF] * (L+2)
for c, d in CD:
    li[c] = d
st = SparseTable(li, max)

def check(m):
    res = 0
    for a, b in AB:
        res = max(res, max(st.get(min(a+m, L+1), L+2)-b+1, 0))
    return res == y

ans = INF
x = 0
while x < L+2:
    y = 0
    for a, b in AB:
        y = max(y, max(st.get(min(a+x, L+1), L+2)-b+1, 0))
    ans = min(ans, x+y)
    x = bisearch_max(x, L+2, check) + 1
print(ans)
