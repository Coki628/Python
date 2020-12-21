"""
・さくっと自力AC
・文字列操作
・pythonはint<->str変換が楽だしzfillとかもあるから、こういうのはやりやすいよね。
・とはいえ、かなり雑に型変換とかスライスをやっているためか、
　pypyでギリギリAC1.89秒、pythonは試しにやったらTLEだった。。
・一応、計算量が10万文字列にクエリ20で最大200万はある想定だから、まあ文字列操作としてはでかいんだけどね。
"""

import sys
from string import digits

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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    S = input()
    S = 'x' + S
    N = len(S)

    j = -1
    for i in range(N-1, -1, -1):
        if j == -1 and S[i] in digits:
            j = i + 1
        if j != -1 and S[i] not in digits:
            T = S[i+1:j]
            if T:
                M = len(T)
                a = int(T)
                a += 1
                T = str(a).zfill(M)
                S = S[:i+1] + T + S[j:]
            break
    print(S[1:])
