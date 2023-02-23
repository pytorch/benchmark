import matplotlib.pyplot as plt
import numpy as np


labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
boys_means = [10,15,10,30,20]
women_means = [25, 32, 34, 20, 25]
girls_means = [10,15,10,30,20]
patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
print(x)
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width-0.1, label='Men', color='b',hatch=patterns[0])
rects2 = ax.bar(x + width/2, women_means, width-0.1, color='b',hatch=patterns[0])
rects3 = ax.bar(x - width/2, boys_means, bottom=men_means,width=width-0.1,color='g',hatch=patterns[1])
rects4 = ax.bar(x + width/2, girls_means, bottom=women_means, width=width-0.1, label='Boys', color='g',hatch=patterns[1])
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
# ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)
ax.bar_label(rects4, padding=3)

fig.tight_layout()
plt.savefig('test_stack')