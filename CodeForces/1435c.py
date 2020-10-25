"""
参考：https://twitter.com/kotatsugame_t/status/1320364873092481025
・自力ならず。
・ソートして尺取
・人によって色々方針あるっぽくて、setで管理してりプリキューで管理したりしてたけど、
　このこたつがめさんの一言が一番しっくりきた。
・尺取すごい久しぶりにやった。最近にぶたんでいけるとこはだいたいにぶたんでやってしまってたからな。。
・全6弦について、全ての音Nを出すためのフレット位置b-aはN*6で全列挙できる。
　これをソートして小さい方から見ていって尺取、見ている区間が常に全てのaを含むように進めていく。
　区間が条件を満たす間chmin取っていけば答えを得られる。
・多分ボトルネックがN*6=60万のソート。尺取は線形だし定数倍あってもさすがにこっちじゃないだろう。
　pypyTLE。C++でAC確認。
・思い出したので修正。今回はソートのキー1つでいいんだからitemgetter使えばよかった。
　そしたら無事pypyAC1.56秒。やっぱこれだけで全然変わるわー。
"""

import sys
from operator import itemgetter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

A = LIST()
M = 6
N = INT()
B = LIST()

li = []
for i in range(M):
    for j in range(N):
        li.append((B[j]-A[i], j))

li.sort(key=itemgetter(0))
L = len(li)
l = r = 0
ans = INF
C = [0] * N
cnt = 0
while l < L:
    while r < L and cnt < N:
        if C[li[r][1]] == 0:
            cnt += 1
        C[li[r][1]] += 1
        r += 1
    if cnt == N:
        ans = min(ans, li[r-1][0] - li[l][0])
    if l == r:
        r += 1
    else:
        C[li[l][1]] -= 1
        if C[li[l][1]] == 0:
            cnt -= 1
    l += 1
print(ans)
