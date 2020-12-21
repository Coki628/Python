"""
参考：https://twitter.com/laycrs/status/1272930971717824513
・自力ならず。。
・逆操作
・終了から遡る。アルファベット昇順に順操作を考えると、直前に操作した所が0でなくなるので、
　アルファベット逆順で逆操作すれば、0になった場所が次に操作するべき場所だと分かる。
・解法分かってからも意外と実装ハマった。同時に0になる場所が複数ある場合もあるので、
　その時にはその0の数だけ処理する必要がある。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    S = input()
    M = len(S)
    N = INT()
    A = LIST()

    # 同時に0になる場所毎に逆操作を行う
    cur = []
    for i, a in enumerate(A):
        if a == 0:
            cur.append(i)
    li = []
    while cur:
        li.append(cur)
        nxt = []
        for i in cur:
            j = i + 1
            while j < N:
                A[j] -= j - i
                if A[j] == 0:
                    nxt.append(j)
                j += 1
            j = i - 1
            while j >= 0:
                A[j] -= i - j
                if A[j] == 0:
                    nxt.append(j)
                j -= 1
        cur = nxt

    # アルファベット大きい方から見る
    C = sorted(Counter(S).items(), reverse=True)
    ans = [''] * N
    i = 0
    for k, v in C:
        # 数が足りなければこの文字は使わない
        if len(li[i]) > v:
            continue
        # 足りれば予め調べた場所に文字を割り当てる
        for j in li[i]:
            ans[j] = k
        i += 1
        if i >= len(li):
            break
    print(''.join(ans))
