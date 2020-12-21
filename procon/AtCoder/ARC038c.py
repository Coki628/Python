"""
・蟻本演習4-2-6
・Grundy数、対称戦略
・これは部分点だけどここまで自力。まあ、ノーヒントでGrundy数気付けたか分からないけど。。
・あと同じ山がたくさんある時は対称戦略で元に戻せるから偶奇だけ分かればOK。
　この考察はゲームでよく出るね。ちゃんと気付けてよかった。
・残るはmex取得パートを高速化するっぽい感じなので、ググる。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
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

N = INT()
A = [0] * N
C = [0] * N
for i in range(1, N):
    C[i], A[i] = MAP()

grundy = [0] * N
grundy[0] = 0
for i in range(1, N):
    se = set()
    # 各茶碗から移動できる範囲で遷移
    for j in range(1, C[i]+1):
        if i-j >= 0:
            se.add(grundy[i-j])
    g = 0
    while g in se:
        g += 1
    grundy[i] = g

nim = 0
for i in range(1, N):
    # 同じ山がたくさんあっても対称戦略で同じ状態に戻せるので偶奇だけ分かればいい
    nim ^= grundy[i]*(A[i]%2)
if nim:
    print('First')
else:
    print('Second')
