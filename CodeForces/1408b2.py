"""
・TLEコードを元に、種類数だけを持つようにする。無事AC。
・これをさらに考察進めると、みんながやってた算数にやつになるね。
"""

import sys

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

for _ in range(INT()):
    N, K = MAP()
    A = LIST()

    ans = INF
    for i in range(1, N+1):
        # 初期状態は全ての位置で種類数が1
        B = [1] * i
        k = 1
        for j in range(1, N):
            # 差分があったら、この位置の種類数を1つ増やして、増やす位置を1つ進める
            if A[j]-A[j-1] != 0:
                B[k%i] += 1
                k += 1
        mx = 0
        for k in range(i):
            mx = max(mx, B[k])
        if mx <= K:
            ans = min(ans, i)
            break
    if ans == INF:
        print(-1)
    else:
        print(ans)
