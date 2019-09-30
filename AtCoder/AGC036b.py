# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/agc036/editorial.pdf
　　　https://www.youtube.com/watch?v=eFQGwwdAVWg
　　　http://kazune-lab.net/contest/2019/07/21/agc036/
・ダブリング
・2^k回後のmod位置じゃなくて、今回は実際の位置(移動距離)を値に格納する。
・これはシミュレーションうまくできてなくてWA。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N, K = MAP()
A = LIST()

# 各値毎の出現位置
B = [[] for i in range(max(A)+1)]
for i, a in enumerate(A+A):
    B[a].append(i)
# 同値が何個先で出現するかを保持する
C = [0] * N
for li in B:
    for i in range(len(li)-1):
        if li[i] >= N:
            break
        C[li[i]] = li[i+1] - li[i]

# nxt[k][i] := iからスタートして、2^kブロック移動後の現在位置
nxt = list2d(61, N, 0)
# 初期化
for i in range(N):
    # iからのスタートは、1(2^0)回目はC[i]+1先に進む
    nxt[0][i] = C[i] + 1
# ダブリングのテーブル構築
for k in range(1, 61):
    for i in range(N):
        nxt[k][i] = nxt[k-1][i] + nxt[k-1][(i+nxt[k-1][i])%N]

# N*K以内でギリギリまで現在位置を進める
def get_cur():
    i = cur = 0
    while True:
        for k in range(61):
            if cur + nxt[k][i] > N * K:
                if k != 0:
                    cur += nxt[k-1][i]
                    i = cur % N
                    break
                else:
                    return cur
cur = get_cur()

# 最後のブロックについてシミュレーションする
ans = []
d = {}
# 開始位置によって切り出し位置を調整
offset = cur
for i in range(cur, N*K):
    a = A[i%N]
    # 追加
    if a not in d:
        ans.append(a)
        d[a] = i
    # 削除
    else:
        s = ans[d[a]-offset:]
        ans = ans[:d[a]-offset]
        # 次回切り出し位置用に調整
        offset += 1
        # 今回切り出して消した分をdictからも削除
        for key in s:
            d.pop(key)
            offset += 1
print(*ans)
