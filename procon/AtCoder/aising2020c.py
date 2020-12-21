"""
・自力ACだけどCにしてはちょっと詰まった。
・枝刈り全探索
・Nまで全部見ると多すぎる。でも。答えが明らかにNを超えるところより先は見る必要ない、
　と考えると大部分は枝刈りできて通る。
・てか後からツイートとか見て分かったけど、枝刈りとかしなくても、
　x^2とかやるんだから、見るの√xとかまででいいね。。
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

C = Counter()
for x in range(1, N+1):
    for y in range(1, N+1):
        if x**2 + y**2 + x*y > N:
            break
        for z in range(1, N+1):
            n = x**2 + y**2 + z**2 + x*y + y*z + z*x
            if n > N:
                break
            C[n] += 1

for i in range(1, N+1):
    print(C[i])
