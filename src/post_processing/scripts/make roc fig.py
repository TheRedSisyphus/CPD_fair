import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
plt.axis('off')

x = np.linspace(0, 5, 10)

ax.plot(x, -3 * x, color='black')
ax.text(x[-3], -3 * x[-3] - 2.2, 'decision boundary', rotation=342, fontsize='large')
ax.plot(x, -3 * x - 10, color='blue')
ax.plot(x, -3 * x + 10, color='red')

ax.add_patch(patches.Polygon(xy=np.array([[0, -10], [5, -25], [5, -30], [0, -30]]), facecolor="blue", alpha=0.2))
ax.add_patch(patches.Polygon(xy=np.array([[0, 10], [5, -5], [5, 15], [0, 15]]), facecolor="red", alpha=0.2))

ax.text(2.5, -5, 'Critical region', fontsize='x-large', color='black')
ax.text(0.1, -29, 'Favored region', fontsize='x-large', color='blue')
ax.text(3.7, 13.5, 'Unfavored region', fontsize='x-large', color='red')

ax.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Privileged', markersize=15, markerfacecolor='grey'),
                   Line2D([0], [0], marker='^', color='w', label='Unprivileged', markersize=15,
                          markerfacecolor='grey')], bbox_to_anchor=(0.9, 0.07))

ax.add_patch(patches.Ellipse((1, -20), 0.2, height=1.8, color='blue'))
ax.add_patch(patches.Ellipse((3, -22), 0.2, height=1.8, color='blue'))
ax.add_patch(patches.Ellipse((2.3, -28), 0.2, height=1.8, color='blue'))
ax.add_patch(patches.Ellipse((3.5, -25), 0.2, height=1.8, color='blue'))
ax.add_patch(patches.Ellipse((0.3, -15), 0.2, height=1.8, color='blue'))

ax.add_patch(patches.Ellipse((1.4, 10), 0.2, height=1.8, color='red'))
ax.add_patch(patches.Ellipse((3.7, 12), 0.2, height=1.8, color='red'))
ax.add_patch(patches.Ellipse((4, 8), 0.2, height=1.8, color='red'))


def get_triangle(x, y, color):
    t_width = 0.2
    t_height = 1.5
    return patches.Polygon(xy=np.array([[x, y], [x + t_width, y], [x + t_width / 2, y + t_height]]), color=color)


ax.add_patch(get_triangle(2, -19, color='blue'))
ax.add_patch(get_triangle(0.5, -24, color='blue'))
ax.add_patch(get_triangle(1.9, -29, color='blue'))

ax.add_patch(get_triangle(4.4, 11, color='red'))
ax.add_patch(get_triangle(4.6, 5, color='red'))
ax.add_patch(get_triangle(3.1, 4, color='red'))
ax.add_patch(get_triangle(2.1, 9.7, color='red'))
ax.add_patch(get_triangle(1.4, 13, color='red'))


# Region critique
ax.add_patch(get_triangle(0.7, 4, color='blue'))
ax.add_patch(get_triangle(1.8, -5, color='blue'))
ax.add_patch(get_triangle(3.6, -14, color='blue'))
ax.add_patch(get_triangle(1.2, -10, color='blue'))

ax.add_patch(patches.Ellipse((0.3, 7.5), 0.2, height=1.8, color='red'))
ax.add_patch(patches.Ellipse((0.9, -10.2), 0.2, height=1.8, color='red'))
ax.add_patch(patches.Ellipse((4.3, -14.9), 0.2, height=1.8, color='red'))
ax.add_patch(patches.Ellipse((3.8, -5.9), 0.2, height=1.8, color='red'))

fig.savefig("/Users/benjamindjian/Desktop/Maîtrise/code/CPDExtract/src/post_processing/scripts/roc.pdf",
            format='pdf',
            dpi=1000)
