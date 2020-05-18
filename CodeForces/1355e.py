"""
参考：各種ツイート
・自力ならず。。
・三分探索
・言われてみれば確かに、だけど、この単調性は考慮できなかったなー。。
・整数の三分探索、前に同値が続くとこでダメになった問題があったから、
　考えたとしてもためらっちゃうんだよな。
・制約1秒でpypyAC0.95秒ギリギリ。まあ10万のNと10^10のlogだからそんなもんか。。
・上限適当に10^10じゃなくてきっちり10^9+1までにしたら、ちょっと良くなって0.86秒。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, ad, rm, mv = MAP()
A = LIST()

def check(m):
    adcnt = rmcnt = 0
    for a in A:
        if a < m:
            adcnt += m - a
        elif a > m:
            rmcnt += a - m
    diff = adcnt - rmcnt
    if diff >= 0:
        return rmcnt*min(ad+rm, mv) + diff*ad
    else:
        return adcnt*min(ad+rm, mv) + -diff*rm

A.sort()
# 三分探索
low = -1
hi = 10**9 + 1
while low+2 < hi:
    m1 = (low*2+hi) // 3
    m2 = (low+hi*2) // 3
    res1 = check(m1)
    res2 = check(m2)
    # 今回は下に凸な関数なので値の小さい方に向かって狭めていく
    if res1 <= res2:
        hi = m2
    else:
        low = m1

ans = min(res1, res2)
print(ans)
