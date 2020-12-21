"""
参考：https://twitter.com/titia_til/status/1323304031796695042
・自力ならず。。
・広義単調減少、広義単調増加を活かして前から貪欲
・まず前から広義単調減少に、できるだけ数を減らす。
　このできるだけ、は、残った数列が広義単調増加を保つ範囲でできる限り、という方針でやる。
・言われたら確かにそれならうまくいくかも、って思えるけど、自分では考え至らなかったなー。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()
    A = LIST()

    cur = A[0]
    A[0] = 0
    for i in range(1, N):
        use = max(min(cur, A[i]-A[i-1]), 0)
        A[i] -= use
        cur = use

    ok = True
    for i in range(1, N):
        if A[i-1] > A[i]:
            ok = False
            break
    if ok:
        YES()
    else:
        NO()
