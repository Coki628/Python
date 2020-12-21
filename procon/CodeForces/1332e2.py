# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/75432
　　　https://twitter.com/titia_til/status/1245020175469965312
　　　https://coonevo.hatenablog.com/entry/2020/04/02/030751
・O(1)解法。
・近いものは考えたけど、使える要素数LRの偶奇による場合分けには気付けなかった。
　(total+1)/2の+1がどうも腑に落ちなくて考えてたけど、使える要素が奇数個だと、
　通り数の合計も奇数個になるので、半分にするなら+1か-1しないといけなくて、
　マス数が偶数の時には、これは必ず「偶奇マスそれぞれが偶数個ずつ」
　(つまりOKの方)の通り数が1つ多くなる。よってMODの数え上げだと、
　(x+1)/2 ってことになる。みたいな感じかなぁ。
・小さいケース手元で試してエスパー、とかが現実的かもしれない。。
・追記。nagaさんが公式の考え方を分かりやすく解説してくれていたので。
・この手の数え上げで、約半分、の約をハッキリさせるために、OK盤面とNG盤面を
　ペアリングするという方法がある。
・下から順にOK,NGをペアリングしていった時、使える要素が奇数個の時は、
　全てのマスが最大値Rを取るような盤面が最終的にペアを作れずに余る。
・よってこの余ってしまった盤面(これはOK盤面)だけ個別に考えて、残りはきれいに半分、とする。
・式にすると、(x-1)/2+1 これは(x+1)/2と同じことになる。
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
INF = 10 ** 18
MOD = 998244353

def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

N, M, L, R = MAP()

NM = N * M
LR = R - L + 1

if NM % 2 == 0:
    # 使える要素数が奇数個の時は半分より1つ多くなる
    if LR % 2 == 1:
        ans = fermat(pow(LR, NM, MOD)+1, 2, MOD)
    else:
        ans = fermat(pow(LR, NM, MOD), 2, MOD)
else:
    # マスの総数が奇数個ならそもそも全部OK
    ans = pow(LR, NM, MOD)
print(ans)
