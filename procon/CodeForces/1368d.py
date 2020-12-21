"""
・ビット系、ANDとOR
・珍しくXORじゃなくてANDとOR。ビット毎に見るのが定石なのは同じだった。
・操作をすると、ANDの側にビットを寄せた感じになって合計のビット数は変わらない。
　最適に操作を続けるとしたら、寄せれるだけ全ビット寄せるのが最適になる。
　(最後にやるのが2乗して和だから、まんべんなくなるより寄せた方がいい)
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
M = 20

bit = [0] * M
for a in A:
    for i in range(M):
        if a>>i & 1:
            bit[i] += 1

B = [0] * N
for i in range(N):
    for j in range(M):
        if bit[j]:
            B[i] += 1<<j
            bit[j] -= 1

ans = 0
for b in B:
    ans += b**2
print(ans)
