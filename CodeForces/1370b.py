"""
・GCDを2にするために、偶奇で分けて偶数同士奇数同士でペアを作る。
・両側ともぴったり偶数あると1つ多くペアが作れてしまうので、それは取り除く。
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
    A = LIST()

    even = []
    odd = []
    for i in range(N*2):
        if A[i] % 2 == 0:
            even.append(i)
        else:
            odd.append(i)
    ans = []
    M1 = len(even)
    M2 = len(odd)
    for i in range(0, M1-(M1&1), 2):
        ans.append((even[i], even[i+1]))
    for i in range(0, M2-(M2&1), 2):
        ans.append((odd[i], odd[i+1]))
    # N-1個にする
    if len(ans) == N:
        ans.pop()
    for a, b in ans:
        print(a+1, b+1)
