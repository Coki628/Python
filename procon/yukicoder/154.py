"""
・自力AC！
・矛盾がないかどうか条件をしっかり作る。こういうの、最初だいたい漏れが出るよね。。
・案の定1ペナ。各時点において、Wが足りてるかのチェックができていなかったので、
　前からの確認部分も追加して無事AC。
・ちなみに64MB制限でpypyがデフォMLEっぽかったのでpythonで通した。
　Counterで100万はやっぱ結構重くて、pythonAC1.43秒。
"""

import sys
from collections import Counter

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
sys.setrecursionlimit(10**9)
INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    S = input()

    C = Counter()
    ok = 1
    for s in S[::-1]:
        C[s] += 1
        if s == 'W' and (C['G'] == 0 or C['R'] == 0):
            ok = 0
            break
        if C['G'] > C['R']:
            ok = 0
            break
    if C['G'] != C['R']:
        ok = 0

    C = Counter()
    for s in S:
        C[s] += 1
        if C['W'] < C['G'] or C['W'] < C['R']:
            ok = 0

    if ok:
        print('possible')
    else:
        print('impossible')
