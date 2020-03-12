# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/joi/2008/2009-yo-prob_and_sol/2009-yo-t3/review/2009-yo-t3-review.html
・自力ならず。Nが1万じゃN^2のループ無理だろ、と愚直実装諦めてしまったけど、
　結果それが想定解だったぽい。
・pythonAC0.07秒って速すぎないか。内側の連鎖のループはたくさん続くことがほぼないって感じなのかな。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST(N)

ans = 0
for i, a in enumerate(A):
    # 3色試す
    for x in range(1, 4):
        if x == a:
            continue
        cnt = 0
        tmp = 1
        j = k = i
        while j-1 >= 0 and k+1 < N:
            # 前が同じ色な限り進める
            while j-1 >= 0 and A[j-1] == x:
                j -= 1
                tmp += 1
            # 後ろが同じ色な限り進める
            while k+1 < N and A[k+1] == x:
                k += 1
                tmp += 1
            # 4以上あれば得点追加
            if tmp >= 4:
                cnt += tmp
                # 次がまだ行けそうなら繰り返す
                if j-1 >= 0 and k+1 < N and A[j-1] == A[k+1]:
                    tmp = 0
                    x = A[j-1]
                else:
                    break
            else:
                break
        ans = max(ans, cnt)
# 残る玉の最小値なので、全体 - 消した数
print(N - ans)
