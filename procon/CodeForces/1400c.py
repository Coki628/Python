"""
参考：https://twitter.com/laycrs/status/1298297994526126085
・自力ならず。。
・逆操作
・かなり近いことしてたのに、あと一歩詰めきれず。。
・0にしなきゃいけないところは決まってて、残りは1埋め。これは合ってた。
・この後の判定パートが違ってて、WとSが逆になったりしてた。。
・正しいのは、S(操作後)が1のところを見て、構築したW(操作前)からそれが作れるか確認。
・こうやって書くとそれはそうだよね、ってなるんだけど、自分が取り組んでる最中は
　どっちがどっちか訳分かんなくなって脳が破滅してたんだよね。。
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
    S = input()
    N = len(S)
    x = INT()

    W = [''] * N
    for i in range(N):
        if S[i] == '0':
            if i-x >= 0:
                W[i-x] = '0'
            if i+x < N:
                W[i+x] = '0'
    for i in range(N):
        if not W[i]:
            W[i] = '1'

    ok = True
    for i in range(N):
        if S[i] == '1':
            if (i-x < 0 or W[i-x] == '0') and (i+x >= N or W[i+x] == '0'):
                ok = False
                break
    if ok:
        ans = ''.join(W)
        print(ans)
    else:
        print(-1)
