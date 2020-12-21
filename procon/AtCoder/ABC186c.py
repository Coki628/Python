"""
・さくっと自力AC
・N進数
・N進数変換はたまに出てくるから、ライブラリ持ってると早くて助かる。
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
sys.setrecursionlimit(10**6)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def dton(num, n):
    res = []
    while abs(num) > 0:
        m = num % abs(n)
        num -= m
        res.append(m)
        num //= n
    res = ''.join(map(str, res))[::-1]
    if res:
        return res
    else:
        return '0'

N = INT()

ans = 0
for i in range(1, N+1):
    i10 = str(i)
    i8 = dton(i, 8)
    if not i10.count('7') and not i8.count('7'):
        ans += 1
print(ans)
