# -*- coding: utf-8 -*-

from itertools import permutations

b11,b12,b13 = map(int, input().split())
b21,b22,b23 = map(int, input().split())
c11,c12 = map(int, input().split())
c21,c22 = map(int, input().split())
c31,c32 = map(int, input().split())

# 3 * 3の各マスを1次配列に格納
# 012
# 345
# 678
# 結果の全パターン出す
p = list(permutations(['o','o','o','o','o','x','x','x','x']))
p = list(set(p))

# 各条件に合わせて加点して、最高得点が出るパターンを割り出す
taka = [0] * len(p)
nao = [0] * len(p)
for i in range(len(p)):
    if p[i][0] == p[i][3]:
        taka[i] += b11
    else:
        nao[i] += b11
    if p[i][1] == p[i][4]:
        taka[i] += b12
    else:
        nao[i] += b12
    if p[i][2] == p[i][5]:
        taka[i] += b13
    else:
        nao[i] += b13
    if p[i][0] == p[i][1]:
        taka[i] += c11
    else:
        nao[i] += c11
    if p[i][1] == p[i][2]:
        taka[i] += c12
    else:
        nao[i] += c12
    if p[i][3] == p[i][4]:
        taka[i] += c21
    else:
        nao[i] += c21
    if p[i][4] == p[i][5]:
        taka[i] += c22
    else:
        nao[i] += c22
    if p[i][6] == p[i][7]:
        taka[i] += c31
    else:
        nao[i] += c31
    if p[i][7] == p[i][8]:
        taka[i] += c32
    else:
        nao[i] += c32

# 最高得点のパターン複数ある…
taka_idx = taka.index(max(taka))
nao_idx = nao.index(max(nao))

# てか、理想形が分かってもどこから置くかとかどーしようかね

print()