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

ans = 0
def check1(x):
    global ans
    if A[x-1] < A[x] > A[x+1]:
        S1.add(x)
        ans += A[x]
        return True
    else:
        return False

def check2(x):
    global ans
    if A[x-1] > A[x] < A[x+1]:
        S2.add(x)
        ans -= A[x]
        return True
    else:
        return False

for _ in range(INT()):
    N, Q = MAP()
    A = [0] + LIST() + [0]
    N += 2

    S1 = set()
    S2 = set()
    ans = 0
    for i in range(1, N-1):
        check1(i)
        check2(i)

    print(ans)
    for _ in range(Q):
        l, r = MAP()
        if l in S1:
            ans -= A[l]
            S1.remove(l)
        if r in S1:
            ans -= A[r]
            S1.remove(r)
        if l in S2:
            ans += A[l]
            S2.remove(l)
        if r in S2:
            ans += A[r]
            S2.remove(r)

        A[l], A[r] = A[r], A[l]

        resl1 = resl2 = False
        if check1(l):
            if l-1 != 0 and l-1 not in S2:
                check2(l-1)
            if l+1 != N-1 and l+1 not in S2:
                resl2 = check2(l+1)
        if check2(l):
            if l-1 != 0 and l-1 not in S1:
                check1(l-1)
            if l+1 != N-1 and l-1 not in S1:
                resl1 = check1(l+1)
        if check1(r):
            if r-1 != 0 and r-1 not in S2:
                check2(r-1)
            if r+1 != N-1 and r+1 not in S2:
                check2(r+1)
            if resl1 and l+1 == r:
                ans -= A[r]
        if check2(r):
            if r-1 != 0 and r-1 not in S1:
                check1(r-1)
            if r+1 != N-1 and r+1 not in S1:
                check1(r+1)
            if resl2 and l+1 == r:
                ans += A[r]
        print(ans)
