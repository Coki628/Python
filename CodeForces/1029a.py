# -*- coding: utf-8 -*-

"""
・結構考えた。Aでこれはなかなか、、って感じ。
"""

import sys
from itertools import accumulate

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

N, K = MAP()
T = input()

# 全部同じ文字は別処理する
if set(T) == 1:
    ans = T + T[0] * (K-1)
    print(ans)
    exit()

# 接頭辞と接尾辞が一致する一番長い所を見つける
i = N - 1
while  T[:i] != T[N-i:]:
    i -= 1
same = T[:i]
diff = T[i:]
# 同じ所は重ねられるので1回でいい
ans = same + diff * K
print(ans)
