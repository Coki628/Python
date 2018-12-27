# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2018/07/22/133200
　　　https://www.hamayanhamayan.com/entry/2018/06/16/091039
・自力ACできたけど、メモをもう一歩深く書けば
　メインのループ1つ落とせるのが分かったのでそっちもやってみる。
→その下処理がかえって重くて(500*500*30=750万)TLEしたからpypyで通した。
　30^4でも81万だから、この場合はメイン側で4重ループ回す方が速いってことだね。
"""
 
N, C = map(int, input().split())
D = [list(map(int, input().split())) for i in range(C)]
cN = [list(map(int, input().split())) for i in range(N)]

# [mod 3でどれになるか][今度の色] = そうなった時の違和感の総和
memo = [[0] * C for i in range(3)]
for c in range(C):
    for i in range(N):
        for j in range(N):
            memo[(i+j+2)%3][c] += D[cN[i][j]-1][c]

ans = float('inf')
# 変わる色を全組み合わせループ
for i in range(C):
    for j in range(C):
        for k in range(C):
            if i == j or j == k or k == i:
                continue
            # メモで違和感の合計まで出してあるので、ここではループなし
            sm = memo[0][i] + memo[1][j] + memo[2][k]
            ans = min(sm, ans)
print(ans)
