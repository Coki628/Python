"""
・自力AC
・最初赤い方も被っちゃダメだと思ってGCD取りに行ったのは自分だけじゃないみたいだ。
　被っていいので、一番小さい幅に合わせてあとはceilすればOK。
"""

import sys

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

N, M = MAP()
A = [0] + sorted(LIST()) + [N+1]

B = []
for i in range(M+1):
    if A[i+1]-A[i]-1 != 0:
        B.append(A[i+1]-A[i]-1)

if not B:
    print(0)
    exit()

mn = min(B)
ans = 0
for b in B:
    ans += ceil(b, mn)
print(ans)
