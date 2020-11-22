"""
・速攻自力AC。みんなもめっちゃ通してた。
・括弧が2種類、それぞれスタックみたいに左を管理して、右が来たら貪欲に繋げる。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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
    S = input()
    N = len(S)

    stack1 = 0
    stack2 = 0
    ans = 0
    for i in range(N):
        if S[i] == '(':
            stack1 += 1
        elif S[i] == ')':
            if stack1:
                stack1 -= 1
                ans += 1
        elif S[i] == '[':
            stack2 += 1
        else:
            if stack2:
                stack2 -= 1
                ans += 1
    print(ans)
