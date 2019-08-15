# -*- coding: utf-8 -*-

"""
・D自力AC！
・大昔にやって(きっと)全然できなかったやつ、完全に忘れてるので再挑戦。ACできて嬉しい。
・imos
・扱うのが時間なので、末端の調整が抜けてて何度かWAしたけど、最終的にちゃんと解決。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
SE = []
for i in range(N):
    s, e = map(int, input().split('-'))
    SE.append((s, e))

for i in range(N):
    s, e = SE[i]
    # 5区切りで下に丸める
    if s%5 != 0:
        s -= s%5
    # 5区切りで上に丸める
    if e%5 != 0:
        e += 5 - e%5
        # 時間の繰り上がりを調整
        if str(e)[-2:] == '60':
            e += 40
    SE[i] = (s, e)

# imosで区間をまとめる
imos = [0] * 2402
for s, e in SE:
    imos[s] += 1
    imos[e+1] -= 1
imos = list(accumulate(imos))

is_raining = False
ans = []
for i in range(2402):
    if is_raining:
        if imos[i] == 0:
            is_raining = False
            e = i - 1
            ans.append((s, e))
    else:
        if imos[i] > 0:
            is_raining = True
            s = i

for s, e in ans:
    print('{0}-{1}'.format(format(s, '04'), format(e, '04')))
