"""
・素因数分解
・一番素因数の大きいやつを使って、他は1箇所に詰め込む。
・*= k**vを*= k*vってやって合わないなーなんでだろーってタイムロス、、これ前もやった。。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

def factorize(num: int) -> dict:
    from math import sqrt
    from collections import Counter

    d = Counter()
    for i in range(2, int(sqrt(num))+1):
        while num % i == 0:
            num //= i
            d[i] += 1
        if num == 1:
            break
    if num != 1:
        d[num] += 1
    return d

for _ in range(INT()):
    N = INT()

    fact = factorize(N).most_common()
    M = fact[0][1]
    ans = [0] * M
    for i in range(M):
        ans[i] = fact[0][0]
    for k, v in fact[1:]:
        ans[-1] *= k**v
    print(len(ans))
    print(*ans)
