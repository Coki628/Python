# -*- coding: utf-8 -*-

"""
・600点自力AC！参戦中のこれは嬉しい。
・問題の見た目は期待値だけど、実際は回数数えて総和取る感じ。
・N=8くらいまで実験して、規則的に増加するのを見出だせた。
・階乗前計算しない版。pythonは多倍長なので、今回みたいに
　10万の階乗1回だけ、とかなら普通にやってしまっても大丈夫。
　ただし途中のMOD取りとMOD除算は適切にやること。(自戒)
・ちょっとだけ遅くなったけど、pythonAC0.7秒、pypyAC0.4秒。
"""

import sys
from math import factorial

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

N = INT()
A = [0] + LIST()
B = [A[i+1] - A[i] for i in range(N)]

add = factorial(N-1) % MOD
# スライムが区間iを通る合計回数をそれぞれ求める
cnt = [0] * N
for i in range(1, N):
    cnt[i] = cnt[i-1] + add
    cnt[i] %= MOD
    add *= fermat(i ,i+1, MOD)
    add %= MOD

ans = 0
for i in range(1, N):
    # 区間長を掛ければ合計距離
    ans += B[i] * cnt[i]
    ans %= MOD
print(ans)
