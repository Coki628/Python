# -*- coding: utf-8 -*-

"""
・再帰検索を各頂点1回通れば良くした版
"""

import sys
sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))

def upd_dpth(children, depth):
    for i in children:
        nodes[i][1] = depth+1 
        upd_dpth(nodes[i][2], depth+1)

N = INT()

# node: [parent, depth, children]
nodes = [[-1, 0, []] for i in range(N)]

for i in range(N):
    node = LIST()
    # 子の記録
    nodes[node[0]][2].extend(node[2:])
    # 親の更新
    for j in nodes[node[0]][2]:
        nodes[j][0] = node[0]

for i in range(N):
    # 深さの更新
    if nodes[i][0] == -1:
        # 深さを引数に持てばルートから1回回るだけで終わる
        upd_dpth(nodes[i][2], 0)

for i in range(N):
    parent, depth, children = nodes[i]
    if parent == -1:
        _type = 'root'
    elif len(children) == 0:
        _type = 'leaf'
    else:
        _type = 'internal node'
    print('node {node}: parent = {parent}, depth = {depth}, {type}, {children}'.format(
        node=i, parent=parent, depth=depth, type=_type, children=children))
