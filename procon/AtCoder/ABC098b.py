# -*- coding: utf-8 -*-

N = int(input())
S = input()

# 切り出しのパターン数ループ
ans = 0
for i in range(1, N):
    # 文字列の切り出し
    str1 = S[:i]
    str2 = S[i:]
    # 重複を削除したリスト
    list1 = list(set(str1))
    list2 = list(set(str2))
    # 1文字ずつ含まれるか確認してカウント
    tmp = 0
    for s in list1:
        if s in list2:
            tmp += 1
    ans = max(ans, tmp)
# 最大値を出力
print(ans)