"""
・インチキ貪欲
・色々あがいたもののTLEだし、同じやつC++で書いたらWAだった。。
・C++で、セグ木に乗せるやつ書いて無事AC。。
"""

import sys

def input(): return sys.stdin.buffer.readline().strip()
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
        S1[x] = 1
        ans += A[x]
        return True
    else:
        return False

def check2(x):
    global ans
    if A[x-1] > A[x] < A[x+1]:
        S2[x] = 1
        ans -= A[x]
        return True
    else:
        return False

out = []
for _ in range(INT()):
    N, Q = MAP()
    A = [0] + LIST() + [0]
    N += 2

    S1 = [0] * N
    S2 = [0] * N
    ans = 0
    for i in range(1, N-1):
        check1(i)
        check2(i)

    out.append(str(ans))
    for _ in range(Q):
        l, r = MAP()
        if S1[l]:
            ans -= A[l]
            S1[l] = 0
        if S1[r]:
            ans -= A[r]
            S1[r] = 0
        if S2[l]:
            ans += A[l]
            S2[l] = 0
        if S2[r]:
            ans += A[r]
            S2[r] = 0

        A[l], A[r] = A[r], A[l]

        resl1 = resl2 = False
        if check1(l):
            if l-1 != 0 and not S2[l-1]:
                check2(l-1)
            if l+1 != N-1 and not S2[l+1]:
                resl2 = check2(l+1)
        if check2(l):
            if l-1 != 0 and not S1[l-1]:
                check1(l-1)
            if l+1 != N-1 and not S1[l+1]:
                resl1 = check1(l+1)
        if check1(r):
            if r-1 != 0 and not S2[r-1]:
                check2(r-1)
            if r+1 != N-1 and not S2[r+1]:
                check2(r+1)
            if resl1 and l+1 == r:
                ans -= A[r]
        if check2(r):
            if r-1 != 0 and not S1[r-1]:
                check1(r-1)
            if r+1 != N-1 and not S1[r+1]:
                check1(r+1)
            if resl2 and l+1 == r:
                ans += A[r]
        out.append(str(ans))

print('\n'.join(out))
