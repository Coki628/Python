"""
・自力ならず。問題の意図が汲み取れなかった。。
・実装系、シミュレーション
・本番中は何が成功パターンなのか結局分からないまま終わってしまった。。
　終了後さらによくよく読み込んでやっと、次の命令が来るまでの時間で、ってことが分かった。
　命令が成功して目的の位置に到達してるのに成功じゃないの意味分からん、、ってなってた。。
・さて、意味が分かったら、少し悩んだけど実装はまあ作れた。
　ちょっとツイートとかで実装方針のヒントは得ていてしまったけど。。
・先に各時間tでのロボットの位置をシミュレートしちゃっておいて、
　後から改めて成功の範囲に入っているかのチェックをやるってすると頭が混乱しにくかった。
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
    N = INT()
    T = [0] * N
    X = [0] * N
    for i in range(N):
        T[i], X[i] = MAP()
    T.append(INF)

    curx = 0
    remain = 0
    pos = [0] * (N+1)
    d = 0
    target = -1
    for i in range(N):
        if remain <= 0:
            if curx < X[i]:
                d = 1
            else:
                d = -1
            remain = abs(curx-X[i])
            target = X[i]
        addt = T[i+1] - T[i]
        remain -= addt
        if remain <= 0:
            curx = target
        else:
            curx += addt*d
        pos[i+1] = curx

    ans = 0
    for i in range(N):
        if pos[i] <= pos[i+1]:
            l, r = pos[i], pos[i+1]
        else:
            l, r = pos[i+1], pos[i]
        if l <= X[i] <= r:
            ans += 1
    print(ans)
