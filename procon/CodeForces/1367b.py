"""
・パリティ
・偶奇でそれぞれ交換が必要な箇所を数えて、一致すればOK。合わなければ不可。
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
    N = INT()
    A = LIST()

    cnt = [0] * 2
    for i in range(N):
        if i % 2 != A[i] % 2:
            cnt[i%2] += 1
    if cnt[0] != cnt[1]:
        print(-1)
    else:
        ans = cnt[0]
        print(ans)
