"""
・なんとか自力AC、時間かかりすぎ。。
・数列を正、負、0に分けて考える。
・地獄みたいな場合分け実装になった。。絶対こんなゴチャゴチャやんなくていいやつ。。
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
INF = 10**19
MOD = 10**9 + 7

for _ in range(INT()):
    N = INT()
    A = LIST()

    A1 = []
    has0 = False
    for i in range(N):
        if A[i] > 0:
            A1.append((A[i], 1))
        elif A[i] < 0:
            A1.append((abs(A[i]), 0))
        else:
            A1.append((0, 2))
            has0 = True

    A1.sort()
    B = []
    cnt = 0
    lasti0 = lasti1 = -1
    for i in range(5):
        a, t = A1.pop()
        if t == 1:
            lasti1 = i
            B.append(a)
        elif t == 0:
            cnt += 1
            lasti0 = i
            B.append(a)
    B2 = B[:]
    cnt2 = cnt
    A2 = A1[:]
    while A1 and cnt % 2 == 1:
        a, t = A1.pop()
        if t == 1 and lasti0 != -1:
            B.pop(lasti0)
            B.append(a)
            cnt -= 1
    while A2 and cnt2 % 2 == 1:
        a, t = A2.pop()
        if t == 0 and lasti1 != -1:
            B2.pop(lasti1)
            B2.append(a)
            cnt2 -= 1
    ans1 = ans2 = 1
    for b in B:
        ans1 *= b
    if cnt % 2 == 1:
        ans1 = -ans1
    if len(B) < 5:
        ans1 = 0
    for b in B2:
        ans2 *= b
    if cnt2 % 2 == 1:
        ans2 = -ans2
    if len(B2) < 5:
        ans2 = 0
    ans = max(ans1, ans2)
    if ans < 0 and has0:
        ans = 0
    if ans < 0:
        ans = 1
        A.sort(reverse=1)
        for i in range(5):
            ans *= A[i]
    print(ans)
