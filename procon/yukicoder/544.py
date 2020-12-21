"""
・結構さくっと自力AC
・負数も使えるので、各桁で+1か-1を使って桁に7が出ないようにうまく帳尻を合わせる。
・基本は上にずらして、上にずらすと7になってしまう6と繰り上がってしまう9だけ下にずらせばOK。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7
EPS = 10 ** -10

S = input()

a = b = 0
for k, s in enumerate(S[::-1]):
    num = int(s)
    if num in [6, 9]:
        a += (num-1) * 10**k
        b += 1 * 10**k
    else:
        a += (num+1) * 10**k
        b += -1 * 10**k
print(a, b)
