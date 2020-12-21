"""
・自力AC
・組み合わせ
・ソース内コメント参照
"""

import sys
from collections import Counter

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

def nC2(n):
    return n*(n-1) // 2

A = LIST(26)
B = {}
for i in range(26):
    c = chr(i+97)
    B[c] = A[i]

C = Counter('helloworld')
ans = 1
for k, v in C.items():
    B[k] -= v
    # そもそも足りないなら0
    if B[k] < 0:
        print(0)
        exit()
    # lとoだけ別処理
    if k == 'l':
        # 2箇所のlでどっちにいくつ割り振るかは全探索する
        ln = B[k]
        mx = 0
        for i in range(ln+1):
            a, b = i, ln - i
            # 前のlは2つ続くので、n個から2個選ぶ、になる
            mx = max(mx, nC2(2+a) * (1+b))
        ans *= mx
    elif k == 'o':
        # 前のoと後ろのoでなるべく均等に割り振るのが最善
        a, b = ceil(B[k], 2), B[k] // 2
        ans *= (1+a) * (1+b)
    else:
        ans *= 1+B[k]
print(ans)
