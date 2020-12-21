"""
・なんとか自力AC
・H,Wが大きいので、全体を数えるのは無理。何とか手持ちの情報から工夫して答えを出す。疲れた。
"""

import sys
from collections import defaultdict

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
row = defaultdict(set)
col = defaultdict(set)
for i in range(N):
    s, k = MAP()
    s -= 1; k -= 1
    row[k].add(s)
    col[s].add(k)

ans = 0
# まず各行から獲得できるマスを取る
for k, se in row.items():
    ans += W - len(se)
# 列側からは行で取らなかったマスだけを取りたいので、取った行数を引く
for s, se in col.items():
    ans += H - len(row)
print(ans)
