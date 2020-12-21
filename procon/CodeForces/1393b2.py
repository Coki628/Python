"""
参考：ツイートとか
・全体はCounterで、8個以上、4個以上、2個以上のものだけsetで個別に管理する。
・これだとそこまで遅くなくてpypyAC0.8秒。
・pypy3のsetのremove遅い説があったため、pypy2でも投げてみたところ、逆に遅くて1.89秒。。
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

N = INT()
A = LIST()
MX = 10 ** 5
C = [0] * (MX+1)
for a in A:
    C[a] += 1

S2 = set()
S4 = set()
S8 = set()
for k, v in enumerate(C):
    if v >= 8:
        S8.add(k)
    elif v >= 4:
        S4.add(k)
    elif v >= 2:
        S2.add(k)

def check():
    if S8:
        # 8
        return True
    if S4:
        if len(S4) >= 2:
            # 4, 4
            return True
        for a in S4:
            if C[a] >= 6 and S2:
                # 6, 2
                return True
            elif len(S2) >= 2:
                # 4, 2, 2
                return True
    return False

for _ in range(INT()):
    op, x = input().split()
    x = int(x)

    if op == '+':
        C[x] += 1
        if C[x] == 8:
            S4.remove(x)
            S8.add(x)
        elif C[x] == 4:
            S2.remove(x)
            S4.add(x)
        elif C[x] == 2:
            S2.add(x)
    else:
        if C[x] == 8:
            S8.remove(x)
            S4.add(x)
        elif C[x] == 4:
            S4.remove(x)
            S2.add(x)
        elif C[x] == 2:
            S2.remove(x)
        C[x] -= 1

    if check():
        YES()
    else:
        NO()
