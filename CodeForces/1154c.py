# -*- coding: utf-8 -*-

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

a, b, c = MAP()

week = [0, 0, 1, 2, 0, 2, 1]
aw = a // 3
bw = b // 2
cw = c // 2
# 1週間分が3つとも足りてる週数分はまとめて進む
mnw = min(aw, bw, cw)
a -= mnw * 3
b -= mnw * 2
c -= mnw * 2

ans = 0
# 最後の週の開始位置を7通り試す
for i in range(7):
    abc = [a, b, c]
    cnt = 0
    for j in range(7):
        day = (i+j) % 7
        if abc[week[day]] > 0:
            abc[week[day]] -= 1
            cnt += 1
        else:
            break
    ans = max(ans, cnt)
# 一番良かったものにまとめて進めた分を足す
ans += mnw * 7
print(ans)
