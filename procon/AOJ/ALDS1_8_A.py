# -*- coding: utf-8 -*-

import sys
sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))

from collections import defaultdict

class Node:
    def __init__(self, key):
        self.key = key
        self.par = -1
        self.left = -1
        self.right = -1

root = None
nodes = defaultdict(lambda: None)

def insert(z):
    global root
    # yはxの親
    y = None
    x = root
    # zの親となるべきノードを探す
    while x is not None:
        y = x
        if z.key < x.key:
            # 左の子に移動
            x = nodes[x.left]
        else:
            # 右の子に移動
            x = nodes[x.right]
    if y is None:
        z.par = -1
    else:
        z.par = y.key
    
    if y is None:
        # Tが空の場合
        root = z
    elif z.key < y.key:
        # zをyの左の子にする
        y.left = z.key
    else:
        # zをyの右の子にする
        y.right = z.key

def tree_walk(node):
    preo.append(node.key)
    if node.left != -1:
        tree_walk(nodes[node.left])
    ino.append(node.key)
    if node.right != -1:
        tree_walk(nodes[node.right])
    return

N = INT()

for _ in range(N):
    instr = input()
    if instr == 'print':
        preo = []
        ino = []
        tree_walk(root)
        print('', *ino)
        print('', *preo)
    else:
        _, key = instr.split()
        key = int(key)
        node = Node(key)
        nodes[key] = node
        insert(node)
