# -*- coding: utf-8 -*-

"""
・座標圧縮
・2次元累積和
・自力AC！
・xとyの添字がこんがらがってなかなかACしなかった。
　ちゃんとケアレスミスなくコード書けるようになりたいね。
"""

N, K = map(int, input().split())

X, Y = [0]*N, [0]*N
for i in range(N):
    X[i], Y[i] = map(int, input().split())

# 座標圧縮：圧縮前の値⇔圧縮後のindexを相互に変換できるようにしておく。
zip_x, zip_y, unzip_x, unzip_y = {}, {}, {}, {}
X2 = sorted(set(X))
Y2 = sorted(set(Y))
# この後の処理のために添字は1-indexedにする
for i in range(len(X2)):
    zip_x[X2[i]] = i+1
    unzip_x[i+1] = X2[i]
for i in range(len(Y2)):
    zip_y[Y2[i]] = i+1
    unzip_y[i+1] = Y2[i]

# 圧縮した座標に点を割り当て(グリッドは累積和用に一回り大きく作る)
grid = [[0] * (len(X2)+1) for _ in range(len(Y2)+1)]
for i in range(N):
    grid[zip_y[Y[i]]][zip_x[X[i]]] += 1

# 2次元累積和
for i in range(len(Y2)+1):
    for j in range(1, len(X2)+1):
        grid[i][j] += grid[i][j-1]
for i in range(1, len(Y2)+1):
    for j in range(len(X2)+1):
        grid[i][j] += grid[i-1][j]

ans = float('inf')
for i in range(1, len(Y2)+1):
    for j in range(1, len(X2)+1):
        # 終点がK以上あれば始点を動かし始める
        if grid[i][j] >= K:
            for k in range(i):
                for l in range(j):
                    # 部分累積和を出して、点がK以上あるか確認
                    if grid[i][j] - grid[i][l] - grid[k][j] + grid[k][l] >= K:
                        # 長方形の面積を出して最小を保持
                        ans = min((unzip_x[j]-unzip_x[l+1]) * (unzip_y[i]-unzip_y[k+1]), ans)
print(ans)
