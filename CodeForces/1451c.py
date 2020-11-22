"""
・自力AC！
・無制限のスワップ操作により、自由に位置は変えられるので、各文字の出現数に注目する。
・加算操作はK個ずつで、同じ数同士じゃないとダメなので、下から見て行って、
　数が合わなければ合うまでK個ずつ値を+1していく、とする。
　これで数の合わない場所があったらNG。
・最初Counterで管理したけど、数列が100万で大きいし、種類数は高々26なのでリストにしてリサブ。
　1.1秒くらいだったのが0.6秒くらいになったのでちょっと安心。無事システスも通った。
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
    N, K = MAP()
    S = [ord(s)-97 for s in input()]
    T = [ord(s)-97 for s in input()]

    C1 = [0] * 26
    C2 = [0] * 26
    for i in range(N):
        C1[S[i]] += 1
        C2[T[i]] += 1

    ok = 1
    for c in range(26):
        if C1[c] < C2[c]:
            ok = 0
            break
        while C1[c] > C2[c]:
            C1[c] -= K
            C1[c+1] += K
        if C1[c] != C2[c]:
            ok = 0
            break
    if ok:
        Yes()
    else:
        No()
