"""
・自力AC！
・水diff埋め1発目としては幸先のいいスタートかな。
・色々やってみようとすると、最初と最後以外は離れてる同じやつはくっつけられなくないか？となる。
・なのでそれをやる、やらない、無理、で1,0,-1の場合分けをやればいい。2回以上操作はない。
・こういうの、ガチャガチャやってみて規則性ぽいものを見つけて、からの祈るように提出、しかできないのほんとやだな。
　ちゃんと証明とかできるようになればいいんだけどねー。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def RLE(data):
    """ ランレングス圧縮 """
    from itertools import groupby

    return [(x, len(list(grp))) for x, grp in groupby(data)]

N = INT()
A = LIST()
C = [0] * (N+1)

rle = RLE(A)
ans = 0
if len(rle) >= 2 and rle[0][0] == rle[-1][0]:
    ans += 1
    rle.pop(0)
    k, _ = rle.pop()
    C[k] += 1

for k, ln in rle:
    C[k] += 1
    if C[k] >= 2:
        print(-1)
        break
else:
    print(ans)
