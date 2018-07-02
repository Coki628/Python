# -*- coding: utf-8 -*-

s = input()

ans = ""
for i in range(len(s)):

    if s[i] == "0":
        ans += "0"
    
    elif s[i] == "1":
        ans += "1"

    elif s[i] == "B":
        # 文字列が空ではない確認
        if ans != "":
            # 残っている時のみ削除を行う(最後の1文字以外を切り取る)
            ans = ans[0:-1]

print(ans)