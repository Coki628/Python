# -*- coding: utf-8 -*-

"""
ライブラリ使わないでDFSで全検索する版
"""

N, K = map(int, input().split())
tNK = [] * N
for i in range(N):
    tNK.append(list(map(int, input().split())))

def dfs(numq, value):
    # 質問が終わったら、0かチェックして返却
    if numq == N:     
        return True if value == 0 else False
    # K個の選択肢を回るループ
    for i in range(K):
        # 自分より下層で0が見つかれば即返却(そうでなければ継続)
        if dfs(numq + 1, value ^ tNK[numq][i]):
            return True
    # 最後まで0が見つからなければOK
    return False

if dfs(0, 0):
    print('Found')
else:
    print('Nothing')
