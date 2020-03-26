# -*- coding: utf-8 -*-

"""
・自力ならず。whileの比較条件でミスってた。
・辞書順最小
・貪欲に自分のindexを目指していいと思ったけど、それだと自分より小さい値を抜かしてしまう場合があるぽい。
　(多分、4,2,1,3みたいので、1,4,2,3から3が2を抜かせてしまうとかそういう感じ)
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
    A = [a-1 for a in LIST()]
    atoi = {}
    for i, a in enumerate(A):
        atoi[a] = i
    used = [0] * N
    # 小さい要素から順にやる
    for a in range(N):
        i = atoi[a]
        j = i
        # 前が自分より大きい値で、ここがまだ動かしてない場所である限り、できるだけ前へ進める
        while j > 0 and a < A[j-1] and not used[j]:
            A[j], A[j-1] = A[j-1], A[j]
            atoi[A[j]] += 1
            used[j] = 1
            j -= 1
        atoi[A[j]] = j
    print(*[a+1 for a in A])
