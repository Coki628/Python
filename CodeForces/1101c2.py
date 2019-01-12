# -*- coding: utf-8 -*-

"""
参考：submission:48261047
・やっぱりpythonで通してる人のやつは
　効率良い実装とかすごく参考になるなー。
"""

T = int(input())
for _ in range(T):
    N = int(input())
    # [l, r, index] * N を作ってソート
    ls = sorted([list(map(int, input().split()))+[i] for i in range(N)])
    group = 1
    # これに添字i指定して直接入れるから、元順ソートはする必要ない
    ans = [0] * N
    # 一番左端が小さいセグメントの右端
    mx = ls[0][1]
    for l, r, i in ls:
        # 左端が今までのセグメントの最大右端より大きければグループ2
        if group == 1 and l > mx:
            group = 2
        # まだグループ1なら最大右端を必要に応じて更新する
        else:
            mx = max(r, mx)
        # 実は一度グループ2に入ったら、そこから先は全部グループ2でいい
        ans[i] = group
    # 最後までグループ1だったら分けられなかったってこと
    if group == 1:
        print(-1)
    else:
        print(*ans)
