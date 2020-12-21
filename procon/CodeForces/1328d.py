# -*- coding: utf-8 -*-

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
    A = LIST()
    ans = [0] * N
    cur = 0
    # とりあえず隣だけ見て2値で埋める
    for i in range(1, N):
        if A[i-1] != A[i]:
            cur = 1 - cur
        ans[i] = cur

    # 先頭と末尾が条件に合ってない場合
    if A[-1] != A[0] and ans[-1] == ans[0]:
        for i in range(1, N):
            # 隣り合う同値のところで帳尻を合わせる
            if A[i-1] == A[i]:
                j = i - 1
                while j >= 0:
                    ans[j] = 1 - ans[j]
                    j -= 1
                break
        else:
            # 隣り合う同値がなければ3値目を使う
            ans[-1] = 2

    ans = [a+1 for a in ans]
    print(max(ans))
    print(*ans)
