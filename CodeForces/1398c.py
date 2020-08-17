"""
・ここまで自力AC
・累積和、zero sum ranges系
・これ手早くやれたのは良かった。ABC146Eが類題。この時と考え方はほぼ一緒で、
　区間の長さ分をフラットにしたいので、先に全要素-1する。
　これで区間和が0になるLRのペアを数えるやつに帰着できる。
"""

import sys
from itertools import accumulate
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    A = [a-1 for a in list(map(int, input()))]

    acc = [0] + list(accumulate(A))
    C = Counter(acc)
    ans = 0
    for v in C.values():
        ans += v * (v-1) // 2
    print(ans)
