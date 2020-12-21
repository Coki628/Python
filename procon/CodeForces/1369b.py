"""
・場合分け、貪欲
・実は11110000とかは全部消して1つの1か0にできる。
・それで消せない部分は、最初の1より前の0と最後の0より後の1なので数える。
・両方足せば答え。10が1つもないと何も消せないのでそれは入力そのまま。
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
    S = input()

    if not S.count('10'):
        print(S)
        continue

    rcnt = 0
    for i in range(N-1, -1, -1):
        if S[i] == '0':
            break
        else:
            rcnt += 1
    lcnt = 0
    for i in range(N):
        if S[i] == '1':
            break
        else:
            lcnt += 1
    ans = '0'*lcnt + '0' + '1'*rcnt
    print(ans)
