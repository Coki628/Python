"""
参考：ツイートとか
・円環、ランレングス圧縮
・実は、ざっくり言うと3つ以上続いてる場所を全部切ればいいので、
　RLEでまとめて、だいたい割る3してやればいいっていう。。
・だいたいって言うのは、微調整として、円環なので最初と最後が同じ方ならどっちかにまとめる、
　全員同じだった場合は別途場合分け、は必要なので。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def RLE(data):
    from itertools import groupby

    return [[x, len(list(grp))] for x, grp in groupby(data)]

for _ in range(INT()):
    N = INT()
    S = input()

    rle = RLE(S)
    if len(rle) == 1:
        ans = ceil(N, 3)
        print(ans)
        continue

    if rle[0][0] == rle[-1][0]:
        rle[0][1] += rle[-1][1]
        rle.pop()
    ans = 0
    for k, v in rle:
        ans += v // 3
    print(ans)
