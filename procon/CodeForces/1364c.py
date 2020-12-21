"""
・自力AC
・mex、隣接リストと優先度付きキュー
・mexとして出現させたい値なら、そこまでは使えない。
　よってどこから使えるかを予め調べておいて、使えるようになったらキューに追加。
・使う順番は小さい方優先がいいので、都度優先度付きキューに詰めて貪欲にやる。
　それで使える値がなくなってしまえばNG。最後までいければOK。
・このタイプの優先度付きキューでのシミュレーションは何回かやってるな。
　今回もちゃんと使えてよかった。
"""

import sys
from heapq import heappush, heappop
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

N = INT()
A = LIST()
MAX = 10 ** 6

# この位置以降はaを使ってOK、の位置を確認
ok = [0] * (MAX+1)
for i, a in enumerate(A):
    ok[a] = i + 1
# 使えるようになる位置毎に隣接リストにまとめる
adjli = defaultdict(list)
for a, i in enumerate(ok):
    adjli[i].append(a)

que = []
ans = [0] * N
for i in range(N):
    # 使えるようになった値をキューに詰める
    for a in adjli[i]:
        heappush(que, a)
    # 要素がなくなったらNG
    if not que:
        print(-1)
        exit()
    # 現在の最小値を答えに追加
    ans[i] = heappop(que)
print(*ans)
