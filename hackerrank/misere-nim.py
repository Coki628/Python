"""
参考：https://www.hamayanhamayan.com/entry/2017/02/27/025050
・蟻本演習4-2-4
・Misere Nim
・最後取ると負けなNim。最初sigmaさんの詳細な解説記事読んでもよく分かんなくて、
　はまやんさんの、ようはどうすればいいかが書いてあるの読んでやっと理解した。
・ようは最初から全部の山が1じゃなければ、うまいことやって普通のNimみたいに勝てる。
　全部1だったら山の数の偶奇で最初から決まってる。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()
    A = LIST()

    mx = max(A)
    if mx >= 2:
        nim = 0
        for a in A:
            nim ^= a
        if nim:
            print('First')
        else:
            print('Second')
    else:
        if N % 2 == 0:
            print('First')
        else:
            print('Second')
