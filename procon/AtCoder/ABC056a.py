# -*- coding: utf-8 -*-

a, b = map(str, input().split())

if a == b == "H":
    print("H")
elif a == "H" and b == "D":
    print("D")
elif a == b == "D":
    print("H")
elif a == "D" and b == "H":
    print("D")