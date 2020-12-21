# -*- coding: utf-8 -*-

time_dict = {}

for i in range(0, 601, 5):

	if 0 <= int(str(i)[-2:]) < 60:
		time_dict[i] = False

time_dict[5] = True
time_dict[350] = True

print(time_dict[350])
print(time_dict[330])

print(str("%04d" % 15))