"""
参考：https://www.hamayanhamayan.com/entry/2017/11/11/022828
・インタラクティブ、三分探索
・はまやんさんの記事から。
・さぶたん、同値が来ない保証があるなら、こんな感じで1つずらしでの比較にすれば、
　にぶたんとほぼ同じロジックで目的の位置を求められる。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def trisearch_min(mn, mx, func):
    """ 三分探索 """

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        # 1つだけずらした場所と比較
        if func(mid) <= func(mid+1):
            ok = mid
        else:
            ng = mid
    return ok

def check1(m):
    print('? {0} 0 0'.format(m), flush=1)
    return INT()

def check2(m):
    print('? 0 {0} 0'.format(m), flush=1)
    return INT()

def check3(m):
    print('? 0 0 {0}'.format(m), flush=1)
    return INT()

x = trisearch_min(-107, 107, check1)
y = trisearch_min(-107, 107, check2)
z = trisearch_min(-107, 107, check3)
print('! {0} {1} {2}'.format(x, y, z), flush=1)
