# pissd = [89.1529,
#          79.4020,
#          71.1872]
#
# ssd3d = [88.4257,
#          78.08,
#          76.87]

pissd = [88.41,
         78.41,
         77.10]

ssd3d = [87.71,
         78.08,
         76.87]

# pissd = [str(x) for x in pissd]
# ssd3d = [str(x) for x in ssd3d]

import numpy as np
import matplotlib.pyplot as plt

# size = 5
# x = np.arange(size)
# a = np.random.random(size)
# b = np.random.random(size)
# c = np.random.random(size)
x = 3

a = pissd
b = ssd3d

total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2

x = [u'easy', u'moderate', u'hard']

my_y_ticks = np.arange(70, 91, 5)
plt.yticks(my_y_ticks)
plt.ylim(65, 91)
plt.ylabel('acc(%)')
plt.bar(x, a, width=width, label='ours', align='edge')
plt.bar([i - width/2 for i in range(len(b))], b, width=width, label='without PI-Fusion')
# plt.bar(x + 2 * width, c, width=width, label='c')
plt.legend()
plt.show()
