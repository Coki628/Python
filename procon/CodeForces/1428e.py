"""
参考：https://twitter.com/ngtkana/status/1317491170054787072
　　　https://twitter.com/laycrs/status/1317489670050050048
・自力ならず。。
・優先度付きキューで貪欲
・それは考えた。けど、人参をいくつに分割するか決められないと思って詰まってしまった。
・見方を変える必要があった。人参を分割するより、ウサギをどの人参に割り当てるか、と考える。
　こうすると、一番有力な人参を保持しながら、1匹ずつウサギを割り当てて状態を変えていける。
　ある人参をいっぺんに何個に割るか考えるんじゃなくて、割り方を1つ増やした時との差分を持っておいて、
　毎回1匹割り当てを増やした時にどのくらい得するかを保持しておく。
"""

import sys
from heapq import heapify, heappush, heappop

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N, K = MAP()
A = LIST()

# 長さaの人参をcnt匹のウサギで分け合う時のコスト
def calc(a, cnt):
    d, m = divmod(a, cnt)
    return d**2 * (cnt-m) + (d+1)**2 * m

# まずは各人参にウサギ1匹ずつを割り当てたものとして初期状態を作る
que = []
for a in A:
    if a // 2 == 0:
        que.append((0, 1, a))
    else:
        que.append((calc(a, 2) - calc(a, 1), 1, a))
heapify(que)

# 次の利得が大きいものから、1匹ずつウサギを割り当てていく
for _ in range(K-N):
    val, cnt, a = heappop(que)
    # ウサギへの割り当てが0になるのはNGなので、次がそうなりそうなら優先度最低にして詰め直す
    if a // (cnt+2) == 0:
        heappush(que, (0, cnt+1, a))
    else:
        heappush(que, (calc(a, cnt+2) - calc(a, cnt+1), cnt+1, a))

ans = 0
for _, cnt, a in que:
    ans += calc(a, cnt)
print(ans)
