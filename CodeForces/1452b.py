"""
・なんとか自力AC。大変だった。
　早く通す人は通してたし、絶対簡単なやり方あったんだろうなぁって感じ。。
・それぞれ最大位置までは積まないといけないので、最大までの必要数、みたいなやつで左右から累積和取る。
・自分が最大だと話が変わってくるので、最大2番目でも同じものを作る。
・必要数までに足りるか足りないかで場合分け。
　足りないなら必要数までの数、足りるなら余る分をmod (N-1)でぴったりいくところまで。
・これを全位置で試して一番コストがかかったところが答え。だと思うけどこれほんとか？
→終了後のTLを眺めていたところ、どうやら一番少ないところだけチェックすればよかったらしい。。。
　これむずくないかな。なんならDよりむずいまである。。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

for _ in range(INT()):
    N = INT()
    mod = N - 1
    A = LIST()

    # 最大と最大2番目を取っておく
    mx1 = max(A)
    mxidx = A.index(max(A))
    mx2 = max(max(A[:mxidx]+[0]), max(A[mxidx+1:]+[0]))
    # 最大までの必要数、みたいなやつで左右から累積和取る
    sm1 = [0] * N
    for i in range(N):
        sm1[i] = mx1 - A[i]
    accl1 = list(accumulate(sm1))
    accr1 = list(accumulate(sm1[::-1]))[::-1]
    sm2 = [0] * N
    for i in range(N):
        sm2[i] = max(mx2 - A[i], 0)
    accl2 = list(accumulate(sm2))
    accr2 = list(accumulate(sm2[::-1]))[::-1]

    ans = 0
    for i in range(N):
        sm = 0
        if i != mxidx:
            if i != 0:
                sm += accl1[i-1]
            if i != N-1:
                sm += accr1[i+1]
        else:
            if i != 0:
                sm += accl2[i-1]
            if i != N-1:
                sm += accr2[i+1]
        # 必要数までに足りるか足りないか
        if sm >= A[i]:
            sm -= A[i]
        else:
            sm = -(A[i]-sm) % mod
        ans = max(ans, sm)
    print(ans)
