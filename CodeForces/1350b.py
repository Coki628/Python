"""
・普通に1次元DPだけど、indexがN+1じゃなくてNしかなくて、最初から1初期化するあたりちょっと変則的。
　最初普通に0初期化でN+1を目指してやってたんだけど、途中で、
　自分の位置は最初から買ったことにした方が都合がよさそう、ってなって1初期化にした。
・内側ループは2,4,6,8..3,6,9..って増えてくタイプのやつだから多分logで済んでる。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    A = LIST()

    # dp[i] := 商品iまで見て、購入できる最大数
    dp = [1] * N
    for i, a in enumerate(A):
        j = i
        # indexがi(1-indexed)の倍数の所だけが遷移候補
        while j+i+1 < N:
            # aの大小関係の条件も確認
            if A[j+i+1] > A[i]:
                # j+i+1の商品を買う遷移
                dp[j+i+1] = max(dp[j+i+1], dp[i] + 1)
            j += i+1
    ans = max(dp)
    print(ans)
