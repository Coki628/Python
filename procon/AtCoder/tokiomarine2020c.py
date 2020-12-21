"""
・自力ならず。。これは悔しい。
・ギャグ(案外少なくて済むを見抜くこと)、いもす
・意外と早く収束するのでは、は疑ったんだけど、1000でやったら20ちょっとで、
　10000でやったら戻ってこなくなったので諦めてしまった。。
・終了後、手抜きして2重ループしてたとこいもすに直したら10000でも普通に動いた。
・その実験用ほぼそのままで提出、pypyAC0.4秒。まじで悔しいー。。
・てかそもそも手元で書いて考察悩んでる時間でさっさと実験始めてれば、
　終わるまでには核心に気づけたかもしれないんだよなぁ。色々反省だ。。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7

N, K = MAP()
A = LIST()

cur = A[:]
for k in range(K):
    imos = [0] * (N+1)
    for i in range(N):
        imos[max(i-cur[i], 0)] += 1
        imos[min(i+cur[i]+1, N)] -= 1
    imos = list(accumulate(imos))
    cur = imos[:-1]
    if min(cur) == N:
        break
print(*cur)
