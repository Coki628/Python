# -*- coding: utf-8 -*-

"""
shell sort
"""

N = int(input())
aN = [int(input()) for i in range(N)]

cnt = 0

def insertion_sort(aN, N, g):
    global cnt
    for i in range(g, N):
        tmp = aN[i]
        j = i - g
        while j >= 0 and aN[j] > tmp:
            aN[j+g] = aN[j]
            j = j - g
            # 要素を前に移動した回数
            cnt += 1
        aN[j+g] = tmp

G = []
g = 1
while g <= N:
    G.append(g)
    g = g * 3 + 1
G.reverse()

# shell sort
for g in G:
    insertion_sort(aN, N, g)

print(len(G))
for i in range(len(G)):
    G[i] = str(G[i])
print(' '.join(G))
print(cnt)
for i in range(N):
    print(aN[i])
