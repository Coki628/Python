"""
・*2ってつけ忘れててWAしてた。気づけたからよかったけど、大幅タイムロスつらい。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    a, b = MAP()

    if b < a:
        a, b = b, a
    x = min(b-a, a)
    a -= x
    b -= x * 2
    ans = x
    d1, a = divmod(a, 3)
    d2, b = divmod(b, 3)
    ans += min(d1, d2) * 2
    if a >= 2 and b >= 1 or a >= 1 and b >= 2:
        ans += 1
    print(ans)
