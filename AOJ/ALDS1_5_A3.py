# -*- coding: utf-8 -*-

"""
全探索
・ビットでやる版
・再帰より弱いのか、2^18でTLEだった。
"""

N = int(input())
aN = list(map(int, input().split()))
Q = int(input())
mQ = list(map(int, input().split()))

for i in range(Q):
    # 全組み合わせ2^Nのループ
    for j in range(1 << N):
        sm = 0
        for k in range(N):
            # 各ビットについて、立っていればその位置の値を加算
            if j >> k & 1:
                sm += aN[k]
        if sm == mQ[i]:
            print('yes')
            break
    else:
        print('no')
