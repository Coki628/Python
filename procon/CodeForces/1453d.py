"""
・終了直後にAC。。悔しい。
・これは終わってからきれいに書き直したやつだけど、多分終了3分後くらい？に投げようとした
　スパゲッティなコードもちゃんと通った。。WAから直したの、iの初期化1箇所の差だった。。
・とはいえまあ、この期待値の遷移はエスパーしただけで全くの半信半疑だったし、
　通ったらラッキー、くらいのものではあったんだけどね。。。
・あと、使える大きい数から貪欲に使ってく実装は、whileで下からやったらなんかグチャグチャしちゃって、
　ダブリングの2冪上から順に使えるか見る時みたいに、大きい方からfor回したらスッキリ書けた。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = 100
dp = [0] * (N+1)
dp[1] = 2
for i in range(2, N+1):
    dp[i] = dp[i-1] + 2**i

for _ in range(INT()):
    K = INT()

    ans = [1]
    K -= 2
    for i in range(N, 0, -1):
        if dp[i] <= K:
            d, m = divmod(K, dp[i])
            for _ in range(d):
                ans += [0]*(i-1)
                ans += [1]
            K = m
    if K:
        print(-1)
    else:
        print(len(ans))
        print(*ans)
