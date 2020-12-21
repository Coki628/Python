"""
参考：https://codeforces.com/blog/entry/81161
・自力ならず。こういう典型テク組み合わせはしっかり解いていきたいところ。
・二分探索、優先度付きキュー
・x以上間隔を空けられるか、で二分探索。それはやろうとしてた。
　中身の判定が残念な貪欲とかしかできなかった。
・優先度付きキュー(C++ならsetでも)で、数が多い順に取り出せるように管理。これを操作して戻す時に工夫。
　間隔をx空ける、は操作x回分は対象に入れてはダメな訳で、それらを一時的に保持できるような
　配列を用意する。(簡単のためdeque使ったけど、ちゃんとindex管理すれば普通の配列でも大丈夫だと思う。)
・あとはキューが捌けた時に、一時配列に個数の残ってる要素がないかを判定、でOK。
・にぶたん内でプリキューなので10万にlog2つ、計算量心配だったけどなんとかAC1.29秒。
"""

import sys
from collections import Counter, deque
from heapq import heappush, heappop

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

def bisearch_max(mn, mx, func):
    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

def check(m):
    tmp = deque()
    que = []
    for k, v in C.items():
        heappush(que, (-v, k))
    
    while que:
        v, k = heappop(que)
        v = -v
        v -= 1
        tmp.append((v, k))
        if len(tmp) >= m:
            v, k = tmp.popleft()
            if v:
                heappush(que, (-v, k))
    for v, k in tmp:
        if v:
            return False
    return True

for _ in range(INT()):
    N = INT()
    A = LIST()

    C = Counter(A)
    res = bisearch_max(0, N, check) - 1
    print(res)
