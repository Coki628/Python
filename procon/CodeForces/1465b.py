"""
・自力AC。けど遅すぎ。。
・鳩の巣原理
・愚直で通る。また鳩の巣にやられた。
・割り切りたい数は最大でも1〜9の9個しかないので、これらのLCM(3000以下くらいらしい)
　くらい以上ループ回せば、絶対どっかでOKな値に当たる。
・明らかに通されすぎなので、まさかと思って愚直投げて通った。マジ気付かない。。
・って言ってたらシステス落ちてた。。ええ。。
　テストケース見たら、ほぼ最大の3000近くループ回すところを1000テストケース=300万くらい回ると、
　2秒でも間に合わないらしい。。set(str(n))のset外したら1.36秒で通った。。
　どうせ18桁しかなかったんだし、重複削除とかわざわざやるんじゃなかった。。。
・まあ、文字列で横着しないでちゃんと数値のままやればよかったんだけどね。。
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

def check(n):
    digits = str(n)
    for d in digits:
        d = int(d)
        if not d:
            continue
        if n % d != 0:
            return 0
    return 1

for _ in range(INT()):
    N = INT()

    while 1:
        if check(N):
            print(N)
            break
        else:
            N += 1
