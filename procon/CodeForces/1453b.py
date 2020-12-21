"""
・なんとか自力AC！
・基本は差分の総和で、そこから一番多く引けるところを引く方針。
・端は別処理して、残りは左右見てそこをいじるとどのくらい得するか、を全箇所見る。
・実装ゴチャゴチャしちゃったけど、まあ割と早く解けた方だったからOK。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()
    A = LIST()

    B = [0] * (N-1)
    for i in range(N-1):
        B[i] = abs(A[i]-A[i+1])
    mx = max(B[0], B[-1])
    for i in range(1, N-1):
        if A[i-1]-A[i] < 0 and A[i]-A[i+1] > 0 or A[i-1]-A[i] > 0 and A[i]-A[i+1] < 0:
            mx = max(mx, abs(min(abs(A[i-1]-A[i]), abs(A[i]-A[i+1])))*2)
    ans = sum(B) - mx
    print(ans)
