# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC
・実装だるー、どうしよ、ってなったけど、行き方向と帰り方向で先に変数作ったらうまくまとまった。
・がしかし実は制約で負の方向に進むことはなかった。(当時と同じ問題文見落とし。。)
・でも実装が鬼場合分けにならなかったのは成長だった。
"""

import sys

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

sx, sy, tx, ty = MAP()

x = tx - sx
y = ty - sy

# 進行方向の確定
ans = ''
if x >= 0:
    x1 = 'R'
    x2 = 'L'
else:
    x1 = 'L'
    x2 = 'R'
if y >= 0:
    y1 = 'U'
    y2 = 'D'
else:
    y1 = 'D'
    y2 = 'U'

# 1周目
ans += x1 * x
ans += y1 * y
ans += x2 * x
ans += y2 * y

# 2周目(ひと回り外側を通る)
ans += y2
ans += x1 * (x+1)
ans += y1 * (y+1)
ans += x2
ans += y1
ans += x2 * (x+1)
ans += y2 * (y+1)
ans += x1

print(ans)
