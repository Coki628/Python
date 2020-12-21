# -*- coding: utf-8 -*-

"""
・400点速攻自力AC
・両端に追加、dequeでうまくいく。
"""

import sys
from collections import deque

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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

S = input()

que = deque(S)
is_rev = 0
for _ in range(INT()):
    q = input().split()
    if q[0] == '1':
        is_rev = 1 - is_rev
    else:
        _, a, c = q
        a = int(a)
        if (a == 1) ^ (not is_rev):
            que.append(c)
        else:
            que.appendleft(c)
ans = ''.join(que)
if is_rev:
    ans = ans[::-1]
print(ans)
