# -*- coding: utf-8 -*-

import sys
sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))

def upd_dpth(node, depth):
    par, sib, deg, dep, hei, l, r = nodes[node]
    height = 0
    if l != -1:
        height = max(height, upd_dpth(l, depth+1))
    if r != -1:
        height = max(height, upd_dpth(r, depth+1))
    nodes[node][3] = depth
    nodes[node][4] = height
    return height+1

N = INT()

# node: [parent, sibing, degree, depth, height, l, r]
nodes = [[-1, -1, 0, 0, 0, -1, -1] for i in range(N)]

for i in range(N):
    node, l, r = MAP()
    if l != -1:
        # 次数の更新
        nodes[node][2] += 1
        # 親の更新
        nodes[l][0] = node
        # 兄弟の更新
        nodes[l][1] = r
        # 左の子の更新
        nodes[node][5] = l
    if r != -1:
        nodes[node][2] += 1
        nodes[r][0] = node
        nodes[r][1] = l
        nodes[node][6] = r


for i in range(N):
    # 深さと高さの更新
    if nodes[i][0] == -1:
        upd_dpth(i, 0)

for i in range(N):
    par, sib, deg, dep, hei, l, r = nodes[i]
    if par == -1:
        _type = 'root'
    elif deg == 0:
        _type = 'leaf'
    else:
        _type = 'internal node'
    print('node {node}: parent = {parent}, sibling = {sibling}, \
degree = {degree}, depth = {depth}, height = {height}, {type}'.format(
        node=i, parent=par, sibling=sib, degree=deg, depth=dep, height=hei, type=_type))
