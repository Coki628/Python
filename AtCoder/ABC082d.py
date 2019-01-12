# -*- coding: utf-8 -*-

"""
・幅優先探索
・4次元dictで訪問済メモ
・まあTLEだけども…。
"""

from collections import defaultdict, deque

S = input()
X, Y = map(int, input().split())
# [x][y][現在の移動距離][現在の向き]をメモする
memo = defaultdict(lambda :defaultdict(lambda :defaultdict(lambda :defaultdict(int))))
len_F = S.count('F')
# ゴールまでの距離と進む予定の距離の偶奇が違えば論外
if (X + Y) % 2 != len_F % 2:
    print('No')
    exit()

def bfs():
    que = deque()
    que.append([0,0,0,'r',len_F])
    while len(que) != 0:
        x, y, cur, dirc, rest = que.popleft()
        # 訪問済
        if memo[x][y][cur][dirc] == 1:
            continue
        memo[x][y][cur][dirc] = 1
        # もうゴールに届かない場合
        if rest < abs(X-x) + abs(Y-y):
            continue
        # 最後までいった時
        if cur == len(S):
            if x == X and y == Y:
                break
            else:
                continue
        # まっすぐ
        if S[cur] == 'F':
            if dirc == 'r': x += 1
            elif dirc == 'l': x -= 1
            elif dirc == 'u': y += 1
            elif dirc == 'd': y -= 1
            que.append([x,y,cur+1,dirc,rest-1])
        # 曲がる
        elif S[cur] == 'T':
            if dirc == 'r' or dirc == 'l':
                que.append([x,y,cur+1,'u',rest])
                que.append([x,y,cur+1,'d',rest])
            elif dirc == 'u' or dirc == 'd': 
                que.append([x,y,cur+1,'r',rest])
                que.append([x,y,cur+1,'l',rest])
bfs()
if any(memo[X][Y][len(S)]):
    print('Yes')
else:
    print('No')
