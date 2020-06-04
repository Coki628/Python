"""
参考：https://detail.chiebukuro.yahoo.co.jp/qa/question_detail/q10115079330
・なんとか自力AC
・微分、積分
・ググりゲーだった。普通にやったらx^-1の積分って0除算で死ぬじゃん、ってなって色々調べた。
・その時だけlogxにすればいいらしい。知らなかった。(か忘れていたのかは分からない。)
"""

import sys
from math import log

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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N = INT()
b = INT()
A = list(map(float, input().split()))

# 微分
def calc1(x):
    res = 0
    for a in A:
        res += a * x**(a-1)
    return res

# 積分
def calc2(x):
    res = 0
    for a in A:
        # x^-1の時はlogxになる
        if a+1 == 0:
            res += log(x)
        else:
            res += 1/(a+1) * x**(a+1)
    return res

print(calc1(b))
print(calc2(b))
