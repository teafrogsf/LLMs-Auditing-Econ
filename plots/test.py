import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import brokenaxes

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
a_sales = np.random.randint(450, 550, size=7)
b_sales = np.random.randint(4800, 5200, size=7)
x = np.arange(len(days))

fig = plt.figure(figsize=(8, 6))
bax = brokenaxes(ylims=((0, 1000), (4500, 5500)), hspace=0.05, d=0.005)

# ---- 手动选择在哪个 y 段画 ----
# 下轴画产品A（小值）
bax.axs[0].bar(x - 0.2, a_sales, width=0.4, label="Product A")

# 上轴画产品B（大值）
bax.axs[1].bar(x + 0.2, b_sales, width=0.4, label="Product B")

# ---- 设置公共部分 ----
bax.set_xticks(x)
bax.set_xticklabels(days)
bax.legend()
plt.show()
