"""
・最初に考えてた方針。よくよく考えて直したらこっちも通った。。
・ちょっとややこしめ？なDP
・連続で左に行けないの条件を、なんとなく同じところ反復するのもダメな気になっていたのがそもそもの敗因。
・左に行った後、連続で行けないように耳で状態持たせたりしてたのを取っ払ったらACしたんだよね。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, K, Z = MAP()
    A = LIST()

    # dp[z][i] := z回左に行って、位置iにいる時の最大値
    dp = list2d(Z+1, N+1, -INF)
    dp[0][0] = 0
    for z in range(Z+1):
        for i in range(N):
            # 普通に右に進む
            dp[z][i+1] = max(dp[z][i+1], dp[z][i]+A[i])
            if z+1 <= Z and 0 < i:
                # 右に行って今の位置に戻る
                dp[z+1][i] = max(dp[z+1][i], dp[z][i]+A[i]+A[i-1])
    ans = 0
    for z in range(Z+1):
        for i in range(N+1):
            # 最終的な位置と左に戻った回数が分かれば、合計の移動回数は一意
            k = i + z*2 - 1
            if k == K:
                # ぴったりK回動いた状態から最大を取る
                ans = max(ans, dp[z][i])
    print(ans)
