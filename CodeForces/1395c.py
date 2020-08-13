"""
・ここまで自力AC
・ビット全探索、ビット演算(AND, OR)
・制約を見て、ビットの状態を全部試せる、となる。
・条件を満たすかどうかは、ビット演算がどうなるかを考えると整う。
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

N, M = MAP()
A = LIST()
B = LIST()

ans = INF
for bit in range(1<<9):
    good = 1
    for a in A:
        ok = 0
        for b in B:
            # 今回のbitに全て含まれるような状態をa&bで作れればOK
            if(bit | (a&b)) == bit:
                ok = 1
                break
        if not ok:
            good = 0
            break
    if good:
        ans = min(ans, bit)
print(ans)
