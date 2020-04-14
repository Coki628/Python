# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/pazzle1230/status/1249738896579035136
　　　https://codeforces.com/contest/1335/submission/76551239
　　　https://twitter.com/titia_til/status/1249739414537830403
・自力ならず。
・種類毎に累積和、左端決め打ちで右端を二分探索、枝刈り
・似たような発想はちょっと考えたけど、内側のbを決めるループを合わせると、
　計算量が200*20万*200で死ぬと思ってやらなかった。。
・実際そのままじゃ無理なんだけど、その左端と同じ数を取れる最大の右端、を取りに行く前に、
　左端が変わっても取れる数が同じだった場合、出てくる右端は同じだし左端は狭まってるしで何も得をしない。
　このことを踏まえると、前回と同じ数だった左端はやる必要がない。
・1つめと2つめのループを入れ替えて、その枝刈りをやったらTLEは取れた。
・でも今度はMLE。累積和はaccumulate使う方が速いはずだけど、どうやらメモリは食うらしい。
　200*20万=4000万の配列に256MB制限だと無理だった。
・累積和の処理を普通のループにして、これでAC1.8秒。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

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

MAX = 200
for _ in range(INT()):
    N = INT()
    A = [0] + [a-1 for a in LIST()]

    acc = list2d(MAX, N+1, 0)
    for i, a in enumerate(A[1:], 1):
        acc[a][i] = 1
    for a in range(MAX):
        for i in range(N):
            acc[a][i+1] += acc[a][i]

    ans = 0
    prevx = -1
    for a in range(MAX):
        for l in range(N+1):
            x = acc[a][l]
            # この枝刈りが肝だった。
            if x == prevx:
                continue
            prevx = x
            r = bisearch_max(-1, N+1, lambda m: acc[a][N] - acc[a][m] >= x)
            if l > r:
                continue
            y = 0
            for b in range(MAX):
                y = max(y, acc[b][r] - acc[b][l])
            ans = max(ans, x*2 + y) 
    print(ans)
