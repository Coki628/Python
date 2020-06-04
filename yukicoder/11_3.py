"""
参考：https://twilog.org/meguru_comp/search?word=%E3%80%90yukicoder%20No.11%E3%80%91&ao=a
　　　https://yukicoder.me/submissions/19885
・公式解寄りの方針
・包除原理
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
EPS = 10 ** -10

W = INT()
H = INT()
N = INT()
row = set()
col = set()
for i in range(N):
    s, k = MAP()
    row.add(k)
    col.add(s)

# 取れる行全部 + 取れる列全部 - 行列の重複分 - 手札の分
ans = len(row)*W + len(col)*H - len(row)*len(col) - N
print(ans)
