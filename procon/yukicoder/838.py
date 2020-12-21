"""
・自力AC！
・集計してソート、耳DP
・結局はその位置に単独でいるやつが、左に動くか右に動くか、の2択で、
　その際に、ずっと手前とかがどうなっていたかは問題ではないので、直前の状態だけ持ってDPできる。
・ちょっと遷移がややこしくなったので、一発で通ったのは気持ちいい。
・詳細はソース内コメント。
"""

import sys
from collections import Counter

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

N = INT()
A = LIST()

C = sorted(Counter(A).items())
M = len(C)

# dp0[i] := 位置C[i][0]まで考えて、直前のやつが自分側に動いていない場合の最小値
dp0 = [INF] * (M+1)
# dp1[i] := 位置C[i][0]まで考えて、直前のやつが自分側に動いた場合の最小値
dp1 = [INF] * (M+1)
dp0[0] = 0
for i, (k, v) in enumerate(C):
    # 移動コストがかかるのはその高さにあるのが1つだけの場合のみ
    if v == 1:
        if i != 0:
            # この星を左に動かす
            dp0[i+1] = dp0[i] + abs(k - C[i-1][0])
        if i != M-1:
            # この星を右に動かす
            dp1[i+1] = dp0[i] + abs(k - C[i+1][0])
        # 1つ手前が動いた場合は自分はもう単独じゃないので動かなくていい
        dp0[i+1] = min(dp0[i+1], dp1[i])
    else:
        # 元々単独でないので動かなくていい
        dp0[i+1] = min(dp0[i], dp1[i])
ans = min(dp0[M], dp1[M])
print(ans)
