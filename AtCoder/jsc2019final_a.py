# -*- coding: utf-8 -*-

"""
・自力AC
・しばらく前にちょっと見て、解法出なかったやつだから嬉しい。
・FFT、二分探索
・全ペアの値の出現数を高速に、でFFTが出た。
・そのあとにぶたんも結構ゴニョゴニョしないといけないし、バグってないか心配だったけど無事一発AC。
・これ300点おかしいだろ。。
"""

import sys
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def fft(A, B, L):
    """ 
    高速フーリエ変換(FFT)
        A：出現回数をカウントしたリスト
        B：出現回数をカウントしたリスト
        L：戻り値配列の長さ(max(A)+max(B)より大きい2冪を渡す)
    """
    import numpy as np
    from numpy.fft import rfft, irfft

    # FFT
    res = irfft(rfft(A, L) * rfft(B, L), L)
    # 四捨五入して整数に
    res = np.rint(res).astype(np.int64)
    return res

N, M = MAP()
A = LIST()
B = LIST()

# 出現数カウント
MAXA = max(A)
MAXB = max(B)
C1 = [0] * (MAXA+1)
C2 = [0] * (MAXB+1)
for a in A:
    C1[a] += 1
for b in B:
    C2[b] += 1

# max(A)+max(B)より大きい2冪
L = 1
k = 0
while L <= MAXA + MAXB:
    k += 1
    L = 2**k

# FFTで全ペアの値についての出現数を出す
res = list(fft(C1, C2, L))
# 2回以上出現した値がなかったらNG
if max(res) < 2:
    print(-1)
    exit()

# この値は絶対2回以上出てる、という値を持ってくる
x = res.index(max(res))
# 出力用に元indexを保持
B = [(b, i) for i, b in enumerate(B)]
B.sort()
# にぶたん用配列
B2, _ = zip(*B)
B2 = list(B2)
ans = []
for i, a in enumerate(A):
    if x - a <= 0:
        continue
    idx = bisect_left(B2, x-a)
    if idx == M:
        continue
    res = a + B2[idx]
    # ぴったりxを作るようなペアなら答えに追加
    if res == x:
        if not ans:
            ans += [i, B[idx][1]]
            # 今使った要素を配列Bから除く(O(M)のpopだけど1回だけだから大丈夫)
            B.pop(idx)
            B2.pop(idx)
        else:
            # 2つめのペアを見つけたら完了
            ans += [i, B[idx][1]]
            break
print(*ans)
