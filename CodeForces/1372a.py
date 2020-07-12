"""
・制約的に2重ループいけるし愚直に確かめた。
・実際は全部1でOKっていうこれまたギャグ。。
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

    used = set([1, 3])
    ans = [1, 1, 3, 3]
    for x in range(4, N+1):
        ok = True
        for a in used:
            if x - a in used:
                ok = False
                break
        if ok:
            ans.append(x)
            ans.append(x)
            used.add(x)
        if len(ans) >= N:
            break
    while len(ans) > N:
        ans.pop()
    print(*ans)
