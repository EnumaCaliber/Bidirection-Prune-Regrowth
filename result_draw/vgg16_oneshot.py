import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Dataset 1 (left table)
x1 = [0.96, 0.97, 0.98, 0.99]
y1 = [92.22, 92.02, 91.75, 91.3]

# Dataset 2 (right table)
x2 = [0.9619, 0.9707, 0.98, 0.99]
y2 = [92.54, 92.32, 92.62, 91.59]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x1, y1, 'o-', color='#E05C5C', linewidth=2, markersize=7, label='origin 1')
ax.plot(x2, y2, 's--', color='#4C72B0', linewidth=2, markersize=7, label='regrowth 2')

# Annotate points
for x, y in zip(x1, y1):
    ax.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(-5, 8),
                fontsize=9, color='#E05C5C')

for x, y in zip(x2, y2):
    ax.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(5, 8),
                fontsize=9, color='#4C72B0')

ax.set_xlabel('X Value', fontsize=12)
ax.set_ylabel('Y Value', fontsize=12)
ax.set_title('Comparison of regrowth and origin', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(0.955, 0.995)

plt.tight_layout()
plt.savefig('plot.png', dpi=150)
plt.show()