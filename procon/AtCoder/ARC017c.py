# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-7、さくっと自力AC
・半分全列挙
・まあ、とても近い類題(AGC026c)やったばっかで、こっちのが下位互換ぽいからね。。
・公式解はソートしてにぶたんだったけど、今回はぴったり一致する場合だからdictで大丈夫。
・以上とか以下を数えるなら、にぶたんすることになると思う。
・計算量2^16*16=105万くらい。pythonでも間に合って、AC0.6秒。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N, X = MAP()
W = LIST(N)

W1 = W[:N//2]
W2 = W[N//2:]
N1 = len(W1)
N2 = len(W2)

# 両方で重さ毎に通り数を集計
d1 = Counter()
for i in range(1<<N1):
    w = 0
    for j in range(N1):
        if i>>j & 1:
            w += W1[j]
    d1[w] += 1
d2 = Counter()
for i in range(1<<N2):
    w = 0
    for j in range(N2):
        if i>>j & 1:
            w += W2[j]
    d2[w] += 1

ans = 0
for w1, cnt in d1.items():
    # w1 + w2 = X → w2 = X - w1
    ans += cnt * d2[X-w1]
print(ans)
