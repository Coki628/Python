# -*- coding: utf-8 -*-

import sys
sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))

def tree_walk(node):
    par, sib, deg, dep, hei, l, r = nodes[node]
    preo.append(node)
    if l != -1:
        tree_walk(l)
    ino.append(node)
    if r != -1:
        tree_walk(r)
    posto.append(node)
    return

N = INT()

# node: [parent, sibing, degree, depth, height, l, r]
nodes = [[-1, -1, 0, 0, 0, -1, -1] for i in range(N)]
preo = []
ino = []
posto = []

for i in range(N):
    node, l, r = MAP()
    if l != -1:
        nodes[l][0] = node
        nodes[node][5] = l
    if r != -1:
        nodes[r][0] = node
        nodes[node][6] = r

for i in range(N):
    # 木の巡回
    if nodes[i][0] == -1:
        tree_walk(i)

print('Preorder')
print(' ', end='')
print(*preo)
print('Inorder')
print(' ', end='')
print(*ino)
print('Postorder')
print(' ', end='')
print(*posto)
