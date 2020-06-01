"""
参考：https://img.atcoder.jp/nomura2020/editorial.pdf
　　　https://www.youtube.com/watch?v=NuhtAfH5SCc&feature=youtu.be
・全然自力ならず。。これ本当に水diff前半かよ。。みんな通しすぎだろ。。
・基本方針としては、下からの最大値と上からの最大値を考えて、両方の条件を満たすようにやる。
・でも解法聞いてこれやってもWA取れずで、色々他の人の提出とか比較して、やっと最後の例外処理見つけた。
・もし全体の方針は分かっても、最後の例外処理は自分で見つけられる気がしないなー。。
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
A = LIST()

# 深さ0の例外処理
if N == 0:
    if A[0] == 1:
        print(1)
    else:
        print(-1)
    exit()

# 条件1：Aの各値から、下から見ての最大を考える
cond1 = [0] * (N+1)
cond1[N] = A[N]
for i in range(N-1, -1, -1):
    cond1[i] = cond1[i+1] + A[i]

if min(cond1) <= 0:
    print(-1)
    exit()

# 条件2：根から見ての最大を考える
cond2 = [0] * (N+1)
cond2[0] = 1
for i in range(1, N+1):
    cond2[i] = min(cond1[i], (cond2[i-1]-A[i-1]) * 2)

if min(cond2) <= 0:
    print(-1)
    exit()

# 最大値がAに足りてるかどうか、深さNだけ上のループでチェックできてないのでここで見る
if cond2[N] < A[N]:
    print(-1)
    exit()

ans = sum(cond2)
print(ans)
