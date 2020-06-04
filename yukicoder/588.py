"""
・結構さくっと自力AC
・回文系
・距離も等間隔という制約があるので、ある位置を中心として左右にいくつ一致する文字が
　あるかを見ていく。ある文字を中心とするかある文字と文字の間を中心とするかで
　微妙に処理が変わるので、2つ関数を作ってそれぞれやる形にした。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7
EPS = 10 ** -10

S = input()
N = len(S)

# x番目の文字を中心とする
def check1(x):
    res = 1
    i = x - 1
    j = x + 1
    while i >= 0 and j < N:
        if S[i] == S[j]:
            res += 2
        i -= 1
        j += 1
    return res

# x番目の文字とx+1番目の文字の間を中心とする
def check2(x):
    res = 0
    i = x
    j = x + 1
    while i >= 0 and j < N:
        if S[i] == S[j]:
            res += 2
        i -= 1
        j += 1
    return res

ans = 0
for i in range(N):
    ans = max(ans, check1(i))
    if i != N-1:
        ans = max(ans, check2(i))
print(ans)
