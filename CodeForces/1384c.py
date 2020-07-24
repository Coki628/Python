"""
・自力ならず。みんな通してた。誤読、、、悔しい。
・貪欲
・subsetって書いてあったのに、連続してなきゃダメな気になってた。
・終了後みんなのツイートで察して、連続してなくてもOKな前提でやったら普通にAC。。
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
    S = [ord(c)-97 for c in input()]
    T = [ord(c)-97 for c in input()]

    ng = False
    for i in range(N):
        if S[i] > T[i]:
            ng = True
            break
    if ng:
        print(-1)
        continue

    cnt = 0
    for c in range(20):
        mn = INF
        idx = []
        for i in range(N):
            if S[i] != T[i] and S[i] == c:
                idx.append(i)
                mn = min(mn, T[i])
        if mn != INF:
            for j in idx:
                S[j] = mn
            cnt += 1
    print(cnt)
