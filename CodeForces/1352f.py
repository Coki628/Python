"""
・0000と1111を先に置いて、後は0101を順番に並べる。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    a, b, c = MAP()

    if b == 0 and c == 0:
        ans = '0' * (a+1)
        print(ans)
        continue
    if a == 0 and b == 0:
        ans = '1' * (c+1)
        print(ans)
        continue
    ans = '0' * (a+1) + '1' * (c+1)
    b -= 1
    i = 0
    while b:
        if i % 2 == 0:
            ans += '0'
        else:
            ans += '1'
        i += 1
        b -= 1
    print(ans)
