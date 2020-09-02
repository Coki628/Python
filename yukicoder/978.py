"""
・結構さくっと自力AC！
・DP、式変形、累積和
・公式解頭いいな。方針違ったけどこれも大丈夫のはず。
・まず問題文のシグマがなんかjが0からiまでみたいな形になってるけど、
　ようはjがiからNまで行く普通の2重ループ(同値ペア含む)と同じと考えていい。できるペアは同じ。
・で、2重ループ無理なのでどうにか状態をまとめたい。a*a+a*b+a*c,+b*b+b+c,+c+cみたいに分けると、
　式変形でくくって、a*(a+b+c)+b*(b+c)+c*(c)みたいに、全部「ひとつ*区間和」の形にできる。
・あとは元の数列の累積和取っておけば、1重ループで全部総和すれば答えが出る。
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

N, p = MAP()

if N == 1:
    print(0)
    exit()

A = [0] * N
A[1] = 1
for i in range(2, N):
    A[i] = (A[i-1]*p + A[i-2]) % MOD

acc = [0] * N
acc[-1] = A[-1]
for i in range(N-2, -1, -1):
    acc[i] = (acc[i+1] + A[i]) % MOD

ans = 0
for i in range(N):
    ans += A[i] * acc[i]
    ans %= MOD
print(ans)
