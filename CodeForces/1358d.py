"""
・終了後自力AC
・累積和、二分探索、等差数列の和
・詰まるところ、どこかの月の最終日で終わるのが最善。の考察に行き着くまでに時間がかかり過ぎた。。
・それが分かれば、全ての月について、最終日に終了を決め打ちすれば、開始点と合計はO(1)かlogで出る。
・尺取したそうな見た目だけど、バグらせそうなので累積和とにぶたんで。
・終了から30分弱くらいで書き終えて、システス後AC。実装自体は20分くらいとかだったと思うし、
　これは時間内に通せるべきだったなー。。考察速度を上げたいよね。
"""

import sys
from itertools import accumulate
from bisect import bisect_left

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

def get_sum(a, d, n):
    """ 等差数列の和：(初項a, 公差d, 項数n) """
    return (2*a + (n-1)*d) * n // 2

N, x = MAP()
A = LIST()

# とりあえず1か月で収まるやつは先に処理
mxA = max(A)
if x <= mxA:
    ans = get_sum(mxA-x+1, 1, x)
    print(ans)
    exit()

# 各月の総和
B = [0] * N
for i in range(N):
    B[i] = get_sum(1, 1, A[i])
# 周期なのでやりやすさのために2周作っておく
A = A + A
B = B + B
# 日数・総和の累積和
accA = list(accumulate(A))
accB = list(accumulate(B))

ans = 0
for i in range(N, N*2):
    # まず月iを総取り
    cnt = B[i]
    remain = x - A[i]
    # どの月まで遡れるかを二分探索
    idx = bisect_left(accA, accA[i-1] - remain)
    # そこまでの総取りできる月の分をまとめて追加
    cnt += accB[i-1] - accB[idx]
    remain -= accA[i-1] - accA[idx]
    # 端数部分を追加
    cnt += get_sum(A[idx]-remain+1, 1, remain)
    ans = max(ans, cnt)
print(ans)
