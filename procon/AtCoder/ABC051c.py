# -*- coding: utf-8 -*-

sx,sy,tx,ty = map(int, input().split())

x = tx - sx
y = ty - sy

ans = ""
if x > 0 and y > 0:
    ans += "U"*y + "R"*x + "D"*y + "L"*x
    ans += "L" + "U"*(y+1) + "R"*(x+1) + "D" + "R" + "D"*(y+1) + "L"*(x+1) + "U"
elif x < 0 and y < 0:
    ans += "D"*y + "L"*x + "U"*y + "R"*x
    ans += "R" + "D"*(y+1) + "L"*(x+1) + "U" + "L" + "U"*(y+1) + "R"*(x+1) + "D"
elif x > 0 and y < 0:
    ans += "R"*x + "D"*y + "L"*x + "U"*y
    ans += "U" + "R"*(x+1) + "D"*(y+1) + "L" + "D" + "L"*(x+1) + "U"*(y+1) + "R"
elif x < 0 and y > 0:
    ans += "L"*x + "U"*y + "R"*x + "D"*y
    ans += "D" + "L"*(x+1) + "U"*(y+1) + "R" + "U" + "R"*(x+1) + "D"*(y+1) + "L"

elif x == 0 and y > 0:
    ans += "U"*y + "R" + "D"*y + "L"
    ans += "L" + "U"*y + "R" + "U" + "R"*2 + "D"*(y+2) + "L"*2 + "U"
elif x == 0 and y < 0:
    ans += "D"*y + "L" + "U"*y + "R"
    ans += "R" + "D"*y + "L" + "D" + "L"*2 + "U"*(y+2) + "R"*2 + "D"
elif x > 0 and y == 0:
    ans += "R"*x + "D" + "L"*x + "U"
    ans += "U" + "R"*x + "D" + "R" + "D"*2 + "L"*(x+2) + "U"*2 + "R"
elif x < 0 and y == 0:
    ans += "L"*x + "U" + "R"*x + "D"
    ans += "D" + "L"*x + "U" + "L" + "U"*2 + "R"*(x+2) + "D"*2 + "L"

print(ans)