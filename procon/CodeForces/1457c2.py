"""
・自力AC。でも時間かかりすぎ。。
・MOD累積和
・MOD K毎に累積和を取る。たまに出るやつ。そして添字バグるやつ。。
・今日色々やっててやっと分かったけど、MOD Kでやるってことは、Kずつずれていくんだから、
　開始位置でいつも+1してるところは+Kになるってこと。考えればそれはそうなんだけど。。
・で、区間和取る時もちゃんとKずつずらす。左端で-1とかやらない。。
・多分、この辺の意識が甘かったから今までよくバグらせてたんだと思う。。
・ちょっと整理してコメントも追加した。とにかく1動かすところをK動かすように意識する。
　それがきっちりできればだいぶ見えるようになる。
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
    N, p, K = MAP()
    p -= 1
    S = input()
    x, y = MAP()

    # mod K毎にやるので、N以上でちょうどいいKの倍数まで伸ばす
    ln = ceil(N, K)*K
    # 後ろの方も同じ区間長で見れるように2倍しておく
    sz = ln*2
    # ここで、いつもN+1みたいにやるところをN以上のKの倍数+Kみたいにする
    A = [0] * (sz+K)
    for i in range(N):
        if S[i] == '0':
            A[i] = x

    # ここも、いつもN+1みたいにやるところをN以上のKの倍数+Kみたいにする
    acc = [0] * (sz+K)
    for i in range(sz):
        acc[i+K] += A[i] + acc[i]

    ans = INF
    for i in range(K, K+N-p):
        # acc[i+N] - acc[i-1] -> acc[i+N以上のKの倍数] - acc[i-K]
        ans = min(ans, acc[p+i+ln] - acc[p+i-K] + (i-K)*y)
    print(ans)
