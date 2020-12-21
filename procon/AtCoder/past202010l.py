"""
・時間内間に合わず、、あと30分あったらな。。公開後、無事AC。。
・位置の偶奇によってで一括に増減させる感じにできる。
　元の数列ではなく、差分の数列で値を管理する。つまり差分の数列を偶奇2種類作る。
　この数列には値がいくつあるかのカウントを持たせておく。
・まとめて動く時はそれを管理する変数を増減、
　1つだけ更新する時は、差分の数列は2箇所しか動かないので、実際に個数を更新。
　この時も偶奇によってどっちが減って増えるか場合分け。
・これは時間あれば解けたっぽいよなぁ。前半が全体的に思ったより重かったのと、
　Jのダイクストラで適切なグラフ作るのに時間かかり過ぎた。。
"""

import sys
from collections import Counter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N, Q = MAP()
H = LIST()

C1 = Counter()
C2 = Counter()
for i in range(N-1):
    if i % 2 == 0:
        C1[H[i+1]-H[i]] += 1
    else:
        C2[H[i+1]-H[i]] += 1

c1cnt = c2cnt = 0
for _ in range(Q):
    q = input()
    if q[0] == '1':
        _, v = q.split()
        v = int(v)
        c1cnt -= v
        c2cnt += v
    elif q[0] == '2':
        _, v = q.split()
        v = int(v)
        c1cnt += v
        c2cnt -= v
    else:
        _, u, v = q.split()
        u = int(u)
        v = int(v)
        u -= 1
        if u % 2 == 0:
            if u != 0:
                C2[H[u]-H[u-1]] -= 1
                C2[H[u]-H[u-1]+v] += 1
            if u != N-1:
                C1[H[u+1]-H[u]] -= 1
                C1[H[u+1]-H[u]-v] += 1
        else:
            if u != 0:
                C1[H[u]-H[u-1]] -= 1
                C1[H[u]-H[u-1]+v] += 1
            if u != N-1:
                C2[H[u+1]-H[u]] -= 1
                C2[H[u+1]-H[u]-v] += 1
        H[u] += v
    ans = C1[-c1cnt] + C2[-c2cnt]
    print(ans)
