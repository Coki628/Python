"""
参考：https://mathtrain.jp/polydivision
・なんとか自力AC
・多項式の割り算
・なんかあったなぁこんな感じの式同士で割り算の筆算みたいなことするやつ。
　と思い出しながら、筆算のやり方確認して、シミュレーションするの書いた。疲れた。。
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

N = INT()
A = LIST()

# 割る数のが大きいので割られる数がそのまま余り
if N <= 2:
    print(N)
    print(*A)
    exit()

# 筆算をシミュレーションする
A = A[::-1]
B = [1, 0, -1, 0]
for i in range(N-4+2):
    # 一番大きい次数が0になるように合わせる
    a = A[i]
    cur = [0] * 4
    for j in range(4):
        cur[j] = B[j] * a
    for j in range(4):
        A[i+j] -= cur[j]
ans = []
for i in range(N):
    if A[i] != 0:
        ans = A[i:]
        break
ans = ans[::-1]
if ans:
    print(len(ans)-1)
    print(*ans)
else:
    print(0)
    print(0)
