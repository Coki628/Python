"""
・自力WA
・集計して多い方から、は合ってるんだけど、もう1歩足りなかった。。
"""

import sys
from collections import Counter

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

    C = Counter(A).most_common()
    adjli = []
    for k, v in C:
        for _ in range(v):
            i = 0
            while i < len(adjli):
                if len(adjli[i]) < 3 and k not in adjli[i]:
                    adjli[i].append(k)
                    break
                i += 1
            else:
                adjli.append([k])
    ans = 0
    for i in range(len(adjli)):
        if len(adjli[i]) == 3:
            ans += 1
    print(ans)
