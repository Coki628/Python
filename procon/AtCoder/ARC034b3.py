# -*- coding: utf-8 -*-

"""
・時間かかったけど何とか自力AC！
・桁和系
・各桁毎で変化する値が決まっているので、それをうまく使ってNに合わせられるかチェック。
・基本的に大きい桁は大きい桁でしか動かせないので、大きい桁から決めていく。
・計算量は再帰で18回くらい2択するので2^18=26万くらい。python0.5秒AC。
・文字列の+が遅いような話があったから、listで持って最後にjoinにしたけど速度変わらず。
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

# 各桁で1増える毎に変化する値
A = [2]
for i in range(18):
    a = int('1{0}1'.format('0'*i))
    A.append(a)
A = A[:N]
A = A[::-1]

# i: 今の桁, cnt: Sから0に向けて減らしていく, num: 答えに使う値
ans = []
def rec(i, cnt, num):
    # 0を下回ったらこのパターンはNG
    if cnt < 0:
        return
    # 残りの値から今回見たい桁を取得
    S = str(cnt).zfill(N)
    s = int(S[i])
    # 最後(1の桁)の判定
    if i == N-1:
        # うまくぴったり合わせられれば、答えに追加
        if cnt%2 == 0 and cnt <= 18:
            num += [cnt//2]
            ans.append(int(''.join(map(str, num))))
    else:
        # 基本は自分の桁以上動かせない(例えば100の桁なら101~909)ので、sかs-1を試していく
        rec(i+1, cnt-A[i]*s, num+[s])
        rec(i+1, cnt-A[i]*((s-1)%10), num+[(s-1)%10])
        if i == N-2:
            # 1の桁だけは18まで範囲を取りうるので、10の桁はs-2もやっておく
            rec(i+1, cnt-A[i]*((s-2)%10), num+[(s-2)%10])

rec(0, int(S), [])
ans.sort()
print(len(ans))
[print(a) for a in ans]
