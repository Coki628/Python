# -*- coding: utf-8 -*-

N = int(input())

C_list = [0] * (N-1)
S_list = [0] * (N-1)
F_list = [0] * (N-1)
for i in range(N-1):
    C_list[i], S_list[i], F_list[i] = map(int, input().split())

# 出発駅の数でループ
for i in range(N-1):
    # 次の行の出力のために初期化
    ans = 0
    # ある出発駅から最終駅までのループ
    for j in range(i, N-1):
        # 開始時の待ち時間
        if j == i:
            ans += S_list[j]
        # 次の駅に着いた時、その駅の開始時待ちがまだある場合
        elif ans < S_list[j]:
            ans = S_list[j]
        # 次の駅への移動時間
        ans += C_list[j]
        # 次の出発前の待ち時間(最終駅に着いた時以外)
        if j != N-2:
            if ans % F_list[j+1] != 0:
                ans += F_list[j+1] - ans % F_list[j+1]
    print(ans)
# 出発駅 = 最終駅の出力
print(0)