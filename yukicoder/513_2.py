"""
参考：https://perogram.hateblo.jp/entry/2019/06/13/153459
・インタラクティブ、二分探索
・三分探索を工夫して二分探索にする。
・まず左端から宝までの位置を取っておいて、ある位置〜宝の距離を取った時に、
　「ある位置〜宝の距離 = 左端〜宝 - 左端〜ある位置」
　となっていれば宝はある位置より右にある。(イメージは自分ノートの図参照)
・これを繰り返せば位置が特定できるので、2次元でそれぞれやる。
・それでもWAだったのはコーナーケース。左端位置を取った時に1発で宝を当てる場合を想定してなかった。
　そこも整えたら無事AC。
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
EPS = 10 ** -10

def bisearch_max(mn, mx, func):
    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

print(0, 0, flush=1)
ldist = INT()
if ldist == 0:
    exit()

def check1(x):
    global ldist
    print(x, 0, flush=1)
    res = INT()
    if res == 0:
        exit()
    else:
        return res == abs(ldist - x)

def check2(y):
    global ldist
    print(0, y, flush=1)
    res = INT()
    if res == 0:
        exit()
    else:
        return res == abs(ldist - y)

x = bisearch_max(-1, 100001, check1)
y = bisearch_max(-1, 100001, check2)
print(x, y, flush=1)
res = INT()
