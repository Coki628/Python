# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc116/editorial.pdf
・制約的にDPとかきついから貪欲かなーは思った。
・heapq使ったり、案外考え方近いとこまで来てたから残念。
・実際、交換していきながら、あとmax取るだけだった。
・でもどっちにしても時間かかり過ぎてたからまあしょうがない。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
from operator import itemgetter, mul
from copy import deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def ceil(a, b=1): return int(-(-a // b))
def round(x): return int((x*2+1) // 2)
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, K = MAP()
tdN = [[0]*2 for i in range(N)]
for i in range(N):
    t, d = MAP()
    tdN[i][0] = t
    tdN[i][1] = d

tdN.sort(key=itemgetter(1), reverse=True)
base = 0
bonus = 0
c = Counter()
changes = []
ans = 0
# とりあえずおいしさ上位から初期値を構成
for i in range(K):
    t, d = tdN[i]
    base += d
    if c[t] == 0:
        bonus += 1
    # 各種類の所持数を管理
    c[t] += 1
    # 2つ以上になる種類は交換候補
    if c[t] >= 2:
        heappush(changes, (d, i))
bonus *= bonus
ans = base + bonus

for i in range(K, N):
    t, d = tdN[i]
    # 既に持ってるネタならスキップ
    if c[t] != 0:
        continue
    else:
        # 1つずつ候補から交換してみて、最大値を取る
        if len(changes):
            _, idx = heappop(changes)
            t2, d2 = tdN[idx]
            base -= d2 - d
            bonus = (sqrt(bonus)+1) ** 2
            ans = max(base + bonus, ans)
            # 今回交換した分を更新
            c[t2] -= 1
            c[t] += 1
print(int(ans))
