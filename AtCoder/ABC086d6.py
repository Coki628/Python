# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回の時にバグらせて死んだやつ。無事AC。
・2次元いもす
・普通の2次元累積和でやる時にmod取る兼ね合いで実装ゴチャゴチャしてたやつ
　(0-indexedだと累積和に場合分けが増えて、1-indexedだとmod取る時にややこしくなる)
　を、うまくいもすに落とせれば割とスッキリいけるんじゃないかと試行錯誤してたやつ。
・結果的には、これも点打ち込む時に色々考えなきゃで大変だったけど、
　正しく点が打てた後は、累積和して最大値見るだけなので、勝手が分かればやりやすいかも。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, K = MAP()
mod = K * 2
XY = []
MAX = K * 4
imos = list2d(MAX+1, MAX+1, 0)
for i in range(N):
    x, y, c = input().split()
    x = int(x)
    y = int(y)
    x %= mod
    y %= mod
    if c == 'B':
        x += K
    # K*4全体には、K*2の領域が4つあるので2*2=4回やる
    for a1 in range(2):
        for a2 in range(2):
            # さらに1つのK*2領域内に、塗る場所が2つ
            for b in [0, K]:
                if c == 'B':
                    b = -b
                x1 = x + a1 * mod + b
                y1 = y + a2 * mod + b
                x2 = x1 + K
                y2 = y1 + K
                # 範囲を一律に0~MAXまでに収める
                x1 = max(min(x1, MAX), 0)
                y1 = max(min(y1, MAX), 0)
                x2 = max(min(x2, MAX), 0)
                y2 = max(min(y2, MAX), 0)
                # ここまで位置を揃えたら後は普通の2次元いもす
                imos[x1][y1] += 1
                imos[x2][y2] += 1
                imos[x1][y2] -= 1
                imos[x2][y1] -= 1

for i in range(MAX+1):
    imos[i] = list(accumulate(imos[i]))
for j in range(MAX+1):
    for i in range(MAX):
        imos[i+1][j] += imos[i][j]
ans = 0
for i in range(K*2, K*4+1):
    ans = max(ans, max(imos[i][K*2:K*4+1]))
print(ans)
