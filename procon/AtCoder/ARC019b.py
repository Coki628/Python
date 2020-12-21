# -*- coding: utf-8 -*-

"""
・自力AC！
・回文系
・適切に場合分けしたらうまくいった。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

S = input()
N = len(S)

# 例外処理(1文字はどう変えても回文)
if N == 1:
    print(0)
    exit()

# 回文判定
cnt = 0
for i in range(N//2):
    if S[i] == S[N-i-1]:
        cnt += 1

# 初めから回文
if cnt == N//2:
    if N%2 == 0:
        # どう変えてもいい
        ans = 25 * N
    else:
        # 奇数の時は真ん中以外
        ans = 25 * (N-1)
# あと1箇所で回文になる場合
elif cnt == N//2-1:
    # 該当箇所の2文字だけは制限付きにする
    ans = 24 * 2 + 25 * (N-2)
# あと2箇所以上必要
else:
    # どう変えてもいい
    ans = 25 * N
print(ans)
