# -*- coding: utf-8 -*-

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

N = INT()

if N <= 52:
    # 1文字では繋げないので2文字と交互
    ans = ceil(N, 2) + N // 2 * 2
    print(ans)
else:
    # 2文字以上あれば、うまくすれば全種類の単語を繋げるはず
    ans = cnt = 0
    k = 1
    while cnt + 26**k <= N:
        cnt += 26**k
        ans += 26**k * k
        k += 1
    ans += (N - cnt) * k
    print(ans)
