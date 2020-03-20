# -*- coding: utf-8 -*-

"""
・相当悩んだ末、なんとか時間内に自力ACした。
・5は末尾が5でない限り割り切らないので、最初は5を使う。
・3は桁和が3の倍数にならなければいいので、5を3,6,9..回使いそうな時は3を1つ増やして調整。
・ってこれ書いてて思ったけど、最初だけ5にして後全部3にすれば万事OKなんじゃあ…。
・はい、それでOKでした。だいたいみんなこれと似た感じでやってるぽい。
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
MOD = 10 ** 9 + 7
 
for _ in range(INT()):
    N = INT()
    if N == 1:
        print(-1)
        continue
    ans = '5' + '3'*(N-1)
    print(ans)
