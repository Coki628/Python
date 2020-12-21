# -*- coding: utf-8 -*-

s = input()

ans = []
for i in range(len(s)):

    if s[i] == "0":
        ans.append("0")
    
    elif s[i] == "1":
        ans.append("1")

    elif s[i] == "B":
        # リストが空ではない確認
        if ans:
            # 文字列が残っている時のみ削除を行う
            del ans[-1]

print("".join(ans))