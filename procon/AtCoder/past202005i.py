"""
・実装おつ。疲れた。。色々ガチャガチャやってなんとか通ったけどここは消耗した。
　条件分岐、もうちょいきれいにまとまる気もするけど、まあとりあえず気合い。。
・10万*10万グリッドだったとしても、値が動く場所はせいぜいクエリの回数分くらいしかないので、
　動いた場所だけdictで持って管理する。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7

N = INT()
Q = INT()

rows = {}
cols = {}
reved = 0
for _ in range(Q):
    li = LIST()
    if li[0] == 1:
        a, b = li[1:]
        a -= 1; b -= 1
        if not reved:
            if a not in rows:
                rows[a] = a
            if b not in rows:
                rows[b] = b
            rows[a], rows[b] = rows[b], rows[a]
        else:
            if a not in cols:
                cols[a] = a
            if b not in cols:
                cols[b] = b
            cols[a], cols[b] = cols[b], cols[a]
    elif li[0] == 2:
        a, b = li[1:]
        a -= 1; b -= 1
        if not reved:
            if a not in cols:
                cols[a] = a
            if b not in cols:
                cols[b] = b
            cols[a], cols[b] = cols[b], cols[a]
        else:
            if a not in rows:
                rows[a] = a
            if b not in rows:
                rows[b] = b
            rows[a], rows[b] = rows[b], rows[a]
    elif li[0] == 3:
        reved = 1 - reved
    else:
        a, b = li[1:]
        a -= 1; b -= 1
        if not reved:
            if a not in rows:
                rows[a] = a
            if b not in cols:
                cols[b] = b
            ans = rows[a]*N + cols[b]
        else:
            if a not in cols:
                cols[a] = a
            if b not in rows:
                rows[b] = b
            ans = cols[a] + rows[b]*N
        print(ans)
