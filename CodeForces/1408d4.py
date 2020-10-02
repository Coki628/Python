import sys
from itertools import accumulate

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
acc = [0] * (L+1)
for c, d in CD:
    acc[c] = d
acc = list(accumulate(acc[::-1], max))[::-1]

def check(m):
    res = 0
    for a, b in AB:
        if a+m >= L+1:
            break
        res = max(res, max(acc[a+m]-b+1, 0))
    return res == y

ans = INF
x = 0
while x < L+2:
    y = 0
    for a, b in AB:
        if a+x >= L+1:
            break
        y = max(y, max(acc[a+x]-b+1, 0))
    ans = min(ans, x+y)
    x = bisearch_max(x, L+2, check) + 1
print(ans)
