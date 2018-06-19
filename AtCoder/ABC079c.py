# -*- coding: utf-8 -*-

num_list = list(input())

# 力技すぎるわ…
if int(num_list[0]) + int(num_list[1]) + int(num_list[2]) + int(num_list[3]) == 7:
    print(num_list[0] + "+" + num_list[1] + "+" \
            + num_list[2] + "+" + num_list[3] + "=7")
elif int(num_list[0]) + int(num_list[1]) + int(num_list[2]) - int(num_list[3]) == 7:
    print(num_list[0] + "+" + num_list[1] + "+" \
            + num_list[2] + "-" + num_list[3] + "=7")
elif int(num_list[0]) + int(num_list[1]) - int(num_list[2]) + int(num_list[3]) == 7:
    print(num_list[0] + "+" + num_list[1] + "-" \
            + num_list[2] + "+" + num_list[3] + "=7")
elif int(num_list[0]) + int(num_list[1]) - int(num_list[2]) - int(num_list[3]) == 7:
    print(num_list[0] + "+" + num_list[1] + "-" \
            + num_list[2] + "-" + num_list[3] + "=7")
elif int(num_list[0]) - int(num_list[1]) + int(num_list[2]) + int(num_list[3]) == 7:
    print(num_list[0] + "-" + num_list[1] + "+" \
            + num_list[2] + "+" + num_list[3] + "=7")
elif int(num_list[0]) - int(num_list[1]) + int(num_list[2]) - int(num_list[3]) == 7:
    print(num_list[0] + "-" + num_list[1] + "+" \
            + num_list[2] + "-" + num_list[3] + "=7")
elif int(num_list[0]) - int(num_list[1]) - int(num_list[2]) + int(num_list[3]) == 7:
    print(num_list[0] + "-" + num_list[1] + "-" \
            + num_list[2] + "+" + num_list[3] + "=7")
elif int(num_list[0]) - int(num_list[1]) - int(num_list[2]) - int(num_list[3]) == 7:
    print(num_list[0] + "-" + num_list[1] + "-" \
            + num_list[2] + "-" + num_list[3] + "=7")