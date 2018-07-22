# -*- coding: utf-8 -*-

sx,sy,tx,ty = map(int, input().split())

x = tx - sx
y = ty - sy

# x > 0 and y > 0 のパターンに限定されてたー 問題よく読めば良かったー
ans = ""
ans += "U"*y + "R"*x + "D"*y + "L"*x
ans += "L" + "U"*(y+1) + "R"*(x+1) + "D" + "R" + "D"*(y+1) + "L"*(x+1) + "U"

print(ans)