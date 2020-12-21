"""
・終了後、40分くらい粘って自力AC。。
・累積和、累積max
・コンテスト中はダメなDP書いて破滅。。実はDPってか累積和と累積max取ればいい。
・左に戻る回数zは全探索。左に戻る回数を決めると、最終的にどの位置で終わるかが一意になるので、
　まずはそこまでの累積和。
・で、左に戻る位置は、一番おいしいところを繰り返すのが最善なので、A[i]+A[i+1]を累積maxしておいて、
　取れる範囲内で最大の場所でz回全部取るようにする。
・最初maxとる場所をi-1してたんだけど、最後に右左の順で動けばもう1つ先まで繰り返しに使えるので、
　accmx[i-1]をaccmx[i]にしたら無事AC。
・終了後、いったん気持ちリセットして別のアプローチを考えたらこの累積和方針が出たから、
　もっと早く切り替えられたらよかったね。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, K, Z = MAP()
    A = LIST()

    B = [0] * (N-1)
    for i in range(N-1):
        B[i] = A[i] + A[i+1]
    accmx = list(accumulate(B, max))
    accmx += [accmx[-1]]
    acc = list(accumulate(A))
    ans = 0
    for z in range(Z+1):
        i = K - z*2 
        if i < 0:
            break
        ans = max(ans, acc[i] + accmx[i]*z)
    print(ans)
