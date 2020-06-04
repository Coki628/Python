"""
・自力AC
・最長回文、Manacher
・Manacher使ってみた。この問題は全体がダメなので、半分に切ったそれぞれで最大を出してる。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def Manacher(S):
    """ Manacher(文字iを中心とする最長の回文の半径を返す)：O(N) """

    N = len(S)
    R = [0] * N
    i = j = 0
    while i < N:
        while i-j >= 0 and i+j < N and S[i-j] == S[i+j]:
            j += 1
        R[i] = j
        k = 1
        while i-k >= 0 and i+k < N and k+R[i-k] < j:
            R[i+k] = R[i-k]
            k += 1
        i += k
        j -= k
    return R

def Manacher_even(_S):
    """ Manacher偶数長(文字iとi+1の間を中心とする最長の回文の半径を返す)：O(N) """

    _N = len(_S)
    S = []
    for i in range(_N-1):
        S.append(_S[i])
        S.append('$')
    S.append(_S[-1])
    N = len(S)
    R = [0] * N
    i = j = 0
    while i < N:
        while i-j >= 0 and i+j < N and S[i-j] == S[i+j]:
            j += 1
        R[i] = j
        k = 1
        while i-k >= 0 and i+k < N and k+R[i-k] < j:
            R[i+k] = R[i-k]
            k += 1
        i += k
        j -= k
    res = [0] * (_N-1)
    j = 0
    for i in range(1, N, 2):
        res[j] = R[i] // 2
        j += 1
    return res

# 文字列Sの最長回文の長さを取得する
def get_max(S):
    res1 = Manacher(S)
    res2 = Manacher_even(S)
    res = 0
    for a in res1:
        # Manacherは半径を返すので、文字列長にする
        a = a*2 - 1
        res = max(res, a)
    for a in res2:
        a *= 2
        res = max(res, a)
    return res

S = input()
N = len(S)

ans = 0
for i in range(1, N):
    # 全体にならないように、各所で切る
    ans = max(ans, get_max(S[:i]), get_max(S[i:]))
print(ans)
