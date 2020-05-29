"""
・内側の処理を数列の長さ2で済ませるようにしたらAC0.8秒。
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
INF = 10 ** 10
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

nums = sorted(set(A), reverse=1)
ans = -INF
for num in nums:
    acc = [0] * 2
    accmn = [0] * 2
    mx = -INF
    for i in range(N):
        if A[i] > num:
            A[i] = -INF
        acc[(i+1)&1] = acc[i&1] + A[i]
        accmn[(i+1)&1] = min(accmn[i&1], acc[i&1])
        mx = max(mx, acc[(i+1)&1]-accmn[i&1])        
    ans = max(ans, mx-num)
print(ans)
