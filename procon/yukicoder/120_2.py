"""
参考：https://twilog.org/meguru_comp/search?word=%E3%80%90yukicoder%20No.120%E3%80%91&ao=a
・自力ならず。。
・集計、優先度付きキュー
・要素を同値毎に集計して、多い方から貪欲に取る。
・優先度付きキューって言われたら解法すぐ出たけど、これ自分で思いつかないのダメだなぁ。。
"""

import sys
from collections import Counter
from heapq import heappush, heappop, heapify

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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

for _ in range(INT()):
    N = INT()
    A = LIST()

    C = list(Counter(A).items())
    for i in range(len(C)):
        k, v = C[i]
        C[i] = (-v, k)
    heapify(C)

    ans = 0
    while len(C) >= 3:
        tmp = []
        for _ in range(3):
            tmp.append(heappop(C))
        for v, k in tmp:
            v = -v
            if v >= 2:
                v -= 1
                heappush(C, (-v, k))
        ans += 1
    print(ans)
