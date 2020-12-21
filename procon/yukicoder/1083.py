"""
・自力ならず。。。
・MODの性質、ビット全探索
・完全に考察方針でミスった。順列全探索を高速化したい雰囲気からbitDPに向かってしまった。
　dp[i] := 集合iまで使い終わった時の最大値、みたいなやつ。
　しばらくWAに悩んだあと、その時点での最大値は必ずしも最終的な最大値に繋がるとは限らないと気付く。
　それでさらに色々悩んで、実は各状態が持つ値の数が思った以上に少ないんじゃないかとか言って
　dp[i][j] := 集合iまで使い終わって値がjの時の最大値、とかやって容赦なくTLE。。降参。
・正解はビット全探索。本質は、MOD1回取ると、それより大きい値でMOD取っても何も変わらないこと。
　このことに注目すると、小さい値でMOD取った後に使う大きい値は無意味、つまり「使ってない」
　と同等に考えることができる。ここから、「ある値を使う/使わない」のビット全探索に持っていける。
　しかも使う値は常に降順に使うと考えていい。昇順にすると後ろの値は使ってないと同じになってしまうため。
・こういう頭いい考察ほんと出ないなー。。脳筋典型しか解けないｗ
"""

import sys

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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N, K = MAP()
A = LIST()

A.sort(reverse=1)
ans = 0
for bit in range((1<<N)-1):
    B = []
    for i in range(N):
        if bit>>i & 1:
            B.append(A[i])
    B.append(A[-1])
    x = K
    for b in B:
        x %= b
    ans = max(ans, x)
print(ans)
