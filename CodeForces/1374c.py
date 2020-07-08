"""
・ペアにするのに数が一致しているのは保証されていたので、
　閉じカッコ余りが来たら後ろに回す、を貪欲にやったら通った。
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
    S = input()

    ans = 0
    cntl = cntr = 0
    for s in S:
        if s == '(':
            cntl += 1
        else:
            cntr += 1
        if cntr > cntl:
            cntr -= 1
            ans += 1
    print(ans)