"""
property_checker.py

中文说明：
- 该脚本用于在给定常数与函数 h、g 的情形下，探索在约束
  h(alpha*eta*p)*mu_r - p*g(alpha*eta*p)*beta*mu_l = u' 下，最大化目标
  eta * p * (beta * g(alpha*eta*p) * mu_l - alpha * mu_l) 的 alpha 与 beta 取值。
- 将 alpha 在线性区间 [alpha_min, alpha_max]（默认 [0,1]）上扫描；
  对每个 alpha，通过约束解出隐含的 beta，并施加可行域 beta ∈ [beta_min, L/mu_l]，
  其中 L 默认为 1500，可通过命令行调整。
- h 与 g 可通过表达式字符串自定义，变量名为 x, eta, p（支持 ** 幂运算）。
- 会打印可行网格上的全局最优，并保存：
  1) 2D 静态图：Objective vs alpha、implied beta vs alpha；
  2) 可选 3D 曲线图：在 (alpha, beta, objective) 空间的可行轨迹并高亮最优点；
  3) 可选 3D 表面图：在 (alpha, beta) 网格上的 objective 曲面，并叠加可行曲线；
  4) 可选 alpha-beta 动画 GIF。

使用方式示例：
  python src/utils/property_checker.py --u_prime 3.5 --alpha_min 0 --alpha_max 1 --num_points 2000 --beta_min 1 --L 1500 --h_expr "x/(eta*p)" --g_expr "x**1.2/(eta*p)" --plot3d --plot_surface --animate --gif_name alpha_beta.gif --save_dir src/utils
"""

import os
import math
import argparse
import numpy as np  # 默认可用
import matplotlib.pyplot as plt  # 默认可用
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def h(x: float, eta: float, p: float) -> float:
	"""默认 h：h(x) = x / (eta * p)。用于约束方程的左端第一项。"""
	return x / (eta * p)


def g(x: float, eta: float, p: float) -> float:
	"""默认 g：g(x) = x^1.2 / (eta * p)。用于约束与目标函数中的权重因子。"""
	return math.pow(x, 1.2) / (eta * p)


def _compile_scalar_func(expr: str):
	"""将形如 'x/(eta*p)' 的表达式编译为标量函数 f(x, eta, p)。"""
	code = compile(expr, "<h_g_expr>", "eval")
	def _f(x: float, eta: float, p: float) -> float:
		return float(eval(code, {"__builtins__": {}}, {"x": x, "eta": eta, "p": p, "np": np, "math": math}))
	return _f


def implied_beta(alpha: float, u_prime: float, eta: float, p: float, mu_l: float, mu_r: float) -> float:
	"""
	根据约束： h(alpha*eta*p)*mu_r - p*g(alpha*eta*p)*beta*mu_l = u'
	解得： beta = (alpha*mu_r - u') / (p * g(alpha*eta*p) * mu_l)
	当分母为 0 或非数时，返回 NaN。
	"""
	g_val = g(alpha * eta * p, eta, p)
	denom = p * g_val * mu_l
	if denom == 0 or math.isnan(denom) or math.isinf(denom):
		return float("nan")
	numer = alpha * mu_r - u_prime
	return numer / denom


def objective(alpha: float, beta: float, eta: float, p: float, mu_l: float) -> float:
	"""
	目标函数： eta * p * (beta * g(alpha*eta*p) * mu_l - alpha * mu_l)。
	在给定 alpha 与隐含 beta 的情况下，计算该目标值。
	"""
	return eta * p * (beta * g(alpha * eta * p, eta, p) * mu_l - alpha * mu_l)


# 将 alpha 范围改为 [0, 1] 线性网格，beta 可行域为 [max(beta_min, L/mu_l), +inf)

def explore_alpha_beta(
	eta: float = 0.9,
	p: float = 1e-4,
	mu_l: float = 1000.0,
	mu_r: float = 5.0,
	u_prime: float = 3.5,
	alpha_min: float = 0.0,
	alpha_max: float = 1.0,
	num_points: int = 2000,
	beta_min: float = 1.0,
	beta_max: float | None = None,
	L: float = 1500.0,
	h_expr: str = "x/(eta*p)",
	g_expr: str = "x**1.2/(eta*p)",
	save_dir: str | None = None,
) -> dict:
	"""
	在 alpha ∈ [alpha_min, alpha_max] 上构造线性网格。
	对每个 alpha：
	  1) 根据约束计算隐含 beta；
	  2) 检查可行域：beta ≥ max(beta_min, L/mu_l)（以及可选上界 beta_max）；
	  3) 若可行，计算目标并记录最优点。
	最终返回最优点及相关参数，并输出图像（可选 3D）。
	"""
	# 编译自定义 h/g 表达式
	h_fn = _compile_scalar_func(h_expr)
	g_fn = _compile_scalar_func(g_expr)

	if save_dir is None:
		save_dir = os.path.dirname(__file__)
	os.makedirs(save_dir, exist_ok=True)

	# alpha 线性网格
	alphas_arr = np.linspace(alpha_min, alpha_max, num_points)
	alphas = [float(a) for a in alphas_arr]

	# 计算有效下/上界：beta ∈ [beta_min, L/mu_l]
	effective_beta_max = (L / mu_l) if mu_l != 0 else float("inf")
	if effective_beta_max < beta_min:
		# 若 L/mu_l 低于 beta_min，则无可行点
		effective_beta_max = beta_min

	best_idx = -1
	best_obj = -float("inf")
	best_alpha = float("nan")
	best_beta = float("nan")

	objs: list[float] = []
	betas: list[float] = []
	feas: list[bool] = []

	def implied_beta_local(alpha_val: float) -> float:
		# 一般 h：numer = h(alpha*eta*p)*mu_r - u'
		numer = h_fn(alpha_val * eta * p, eta, p) * mu_r - u_prime
		denom = p * g_fn(alpha_val * eta * p, eta, p) * mu_l
		if denom == 0 or math.isnan(denom) or math.isinf(denom):
			return float("nan")
		return numer / denom

	def objective_local(alpha_val: float, beta_val: float) -> float:
		return eta * p * (beta_val * g_fn(alpha_val * eta * p, eta, p) * mu_l - alpha_val * mu_l)

	for a in alphas:
		b = implied_beta_local(a)
		feasible = True
		if math.isnan(b) or math.isinf(b):
			feasible = False
		if b < beta_min:
			feasible = False
		# 强制上界：min(用户给定 beta_max, L/mu_l)
		user_beta_max = beta_max if beta_max is not None else float("inf")
		beta_upper = min(user_beta_max, effective_beta_max)
		if b > beta_upper:
			feasible = False

		if feasible:
			obj = objective_local(a, b)
		else:
			obj = -float("inf")

		objs.append(obj)
		betas.append(b)
		feas.append(feasible)

		if feasible and obj > best_obj:
			best_obj = obj
			best_idx = len(objs) - 1
			best_alpha = a
			best_beta = b

	# 静态 2D 图：目标与隐含 beta 随 alpha 的变化
	try:
		fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
		ax1, ax2 = axes
		plot_alphas = [a for a, ok in zip(alphas, feas) if ok]
		plot_objs = [o for o, ok in zip(objs, feas) if ok]
		plot_betas = [b for b, ok in zip(betas, feas) if ok]
		if len(plot_alphas) > 0:
			ax1.plot(plot_alphas, plot_objs)
			ax1.set_xlabel("alpha")
			ax1.set_ylabel("Objective")
			ax1.set_title(f"Objective vs alpha (u'={u_prime:g})")
			ax1.grid(True, which="both", ls=":", alpha=0.5)
			# 高亮最优
			if best_idx >= 0 and feas[best_idx]:
				ax1.plot([alphas[best_idx]], [objs[best_idx]], "*", color="tab:red", markersize=10, label="best")
				ax1.legend()

			ax2.plot(plot_alphas, plot_betas, color="tab:orange")
			ax2.set_xlabel("alpha")
			ax2.set_ylabel("beta (implied)")
			ax2.set_title("Implied beta vs alpha")
			ax2.grid(True, which="both", ls=":", alpha=0.5)
			# 高亮最优
			if best_idx >= 0 and feas[best_idx]:
				ax2.plot([alphas[best_idx]], [betas[best_idx]], "*", color="tab:red", markersize=10)
		else:
			ax1.text(0.5, 0.5, "No feasible points", ha="center", va="center", transform=ax1.transAxes)
			ax2.text(0.5, 0.5, "No feasible points", ha="center", va="center", transform=ax2.transAxes)

		out_path = os.path.join(save_dir, "property_checker_alpha_beta_uprime_3p5.png")
		fig.suptitle(f"eta={eta}, p={p}, mu_l={mu_l}, mu_r={mu_r}, L={L}, beta_range=[{beta_min:.3g}, {effective_beta_max:.3g}]")
		fig.savefig(out_path, dpi=150)
		plt.close(fig)
		print(f"已保存图像：{out_path}")
	except Exception as e:  # noqa: BLE001
		print(f"绘图失败，已跳过。原因：{e}")

	return {
		"alpha_star": best_alpha,
		"beta_star": best_beta,
		"obj_star": best_obj,
		"u_prime": u_prime,
		"eta": eta,
		"p": p,
		"mu_l": mu_l,
		"mu_r": mu_r,
		"alphas": alphas,
		"betas": betas,
		"objs": objs,
		"feas": feas,
		"best_idx": best_idx,
		"effective_beta_max": effective_beta_max,
		"h_expr": h_expr,
		"g_expr": g_expr,
		"h_fn": h_fn,  # 返回以便后续绘制
		"g_fn": g_fn,
	}


def plot_3d_curve(alphas: list[float], betas: list[float], objs: list[float], feas: list[bool], best_idx: int, save_path: str) -> None:
	"""
	绘制 3D 三维曲线：(alpha, beta, objective) 的可行轨迹，并高亮全局最优。
	"""
	curve_x = [a for a, ok in zip(alphas, feas) if ok]
	curve_y = [b for b, ok in zip(betas, feas) if ok]
	curve_z = [o for o, ok in zip(objs, feas) if ok]
	if len(curve_x) == 0:
		print("无可行点，跳过 3D 图生成。")
		return

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(curve_x, curve_y, curve_z, color='tab:blue', lw=1.8)
	ax.set_xlabel('alpha')
	ax.set_ylabel('beta')
	ax.set_zlabel('objective')
	ax.set_title('Feasible trajectory in (alpha, beta, objective)')
	# 高亮最优点
	if best_idx >= 0 and 0 <= best_idx < len(alphas) and all([len(alphas)>best_idx, len(betas)>best_idx, len(objs)>best_idx]):
		ax.scatter([alphas[best_idx]], [betas[best_idx]], [objs[best_idx]], color='tab:red', s=50, marker='*', depthshade=True)
		ax.text(alphas[best_idx], betas[best_idx], objs[best_idx], ' best', color='tab:red')

	fig.savefig(save_path, dpi=150)
	plt.close(fig)
	print(f"已保存 3D 图：{save_path}")


def plot_objective_surface(eta: float, p: float, mu_l: float, h_fn, g_fn, alpha_min: float, alpha_max: float, beta_min_eff: float, beta_plot_max: float, grid_n: int, curve_alphas: list[float], curve_betas: list[float], curve_objs: list[float], curve_feas: list[bool], best_idx: int, save_path: str) -> None:
	"""
	绘制 objective(alpha,beta) 的 3D 曲面，并叠加可行轨迹曲线。
	"""
	A = np.linspace(alpha_min, alpha_max, grid_n)
	B = np.linspace(beta_min_eff, beta_plot_max, grid_n)
	AA, BB = np.meshgrid(A, B)
	# 注意：objective 面与 h 无关，仅与 g 有关
	GG = np.zeros_like(AA)
	for i in range(AA.shape[0]):
		for j in range(AA.shape[1]):
			GG[i, j] = eta * p * (BB[i, j] * g_fn(AA[i, j] * eta * p, eta, p) * mu_l - AA[i, j] * mu_l)

	fig = plt.figure(figsize=(9, 7))
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(AA, BB, GG, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
	ax.set_xlabel('alpha')
	ax.set_ylabel('beta')
	ax.set_zlabel('objective')
	ax.set_title('Objective surface with feasible curve')
	fig.colorbar(surf, shrink=0.5, aspect=10)

	# 固定轴范围，避免自动取整到 0 或 100
	ax.set_xlim(alpha_min, alpha_max)
	ax.set_ylim(beta_min_eff, beta_plot_max)
	# 可选：设定更直观的 y 轴刻度
	try:
		yticks = np.linspace(beta_min_eff, beta_plot_max, 6)
		ax.set_yticks(yticks)
	except Exception:
		pass

	# 叠加可行曲线
	curve_x = [a for a, ok in zip(curve_alphas, curve_feas) if ok]
	curve_y = [b for b, ok in zip(curve_betas, curve_feas) if ok]
	curve_z = [o for o, ok in zip(curve_objs, curve_feas) if ok]
	if len(curve_x) > 0:
		ax.plot(curve_x, curve_y, curve_z, color='tab:red', lw=2.0, label='feasible curve')
		ax.legend()
		if best_idx >= 0 and 0 <= best_idx < len(curve_alphas):
			ax.scatter([curve_alphas[best_idx]], [curve_betas[best_idx]], [curve_objs[best_idx]], color='k', marker='*', s=60)

	fig.savefig(save_path, dpi=150)
	plt.close(fig)
	print(f"已保存 Objective 3D 曲面图：{save_path}")


def animate_alpha_beta_curve(alphas: list[float], betas: list[float], feas: list[bool], objs: list[float], save_path: str) -> None:
	"""
	绘制 alpha-beta 同步动画：
	- 背景为可行点的参数曲线 beta(alpha)
	- 动画为一个点沿着可行曲线移动，同时标题显示当前 (alpha, beta, obj)
	"""
	curve_x = [a for a, ok in zip(alphas, feas) if ok]
	curve_y = [b for b, ok in zip(betas, feas) if ok]
	curve_o = [o for o, ok in zip(objs, feas) if ok]
	if len(curve_x) == 0:
		print("无可行点，跳过动画生成。")
		return

	fig, ax = plt.subplots(figsize=(6, 5))
	ax.plot(curve_x, curve_y, color="tab:blue", alpha=0.7)
	ax.set_xlabel("alpha")
	ax.set_ylabel("beta")
	ax.set_title("alpha-beta trajectory")
	ax.grid(True, ls=":", alpha=0.4)

	(point_line,) = ax.plot([], [], "o", color="tab:red")

	def init():
		point_line.set_data([], [])
		return (point_line,)

	def update(frame_idx: int):
		x = curve_x[frame_idx]
		y = curve_y[frame_idx]
		o = curve_o[frame_idx]
		point_line.set_data([x], [y])
		ax.set_title(f"alpha-beta trajectory | alpha={x:.4g}, beta={y:.4g}, obj={o:.4g}")
		return (point_line,)

	anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(curve_x), interval=30, blit=True)
	writer = animation.PillowWriter(fps=30)
	anim.save(save_path, writer=writer)
	plt.close(fig)
	print(f"已保存动画：{save_path}")


def main():
	"""命令行入口：解析参数并调用探索函数，打印最优结果。"""
	parser = argparse.ArgumentParser(description="Property checker: 探索 u'=3.5 时 alpha/beta")
	parser.add_argument("--eta", type=float, default=0.9)
	parser.add_argument("--p", type=float, default=1e-4)
	parser.add_argument("--mu_l", type=float, default=1000.0)
	parser.add_argument("--mu_r", type=float, default=5.0)
	parser.add_argument("--u_prime", type=float, default=3.5)
	parser.add_argument("--alpha_min", type=float, default=0.0)
	parser.add_argument("--alpha_max", type=float, default=1.0)
	parser.add_argument("--num_points", type=int, default=2000)
	parser.add_argument("--beta_min", type=float, default=1.0)
	parser.add_argument("--beta_max", type=float, default=None)
	parser.add_argument("--L", type=float, default=1500.0)
	parser.add_argument("--h_expr", type=str, default="x/(eta*p)")
	parser.add_argument("--g_expr", type=str, default="x**1.2/(eta*p)")
	parser.add_argument("--save_dir", type=str, default=os.path.dirname(__file__))
	parser.add_argument("--animate", action="store_true", help="生成 alpha-beta 同步动画")
	parser.add_argument("--gif_name", type=str, default="alpha_beta.gif")
	parser.add_argument("--plot3d", action="store_true", help="生成 (alpha, beta, obj) 3D 曲线图")
	parser.add_argument("--plot3d_name", type=str, default="alpha_beta_obj_3d.png")
	parser.add_argument("--plot_surface", action="store_true", help="生成 objective(alpha,beta) 3D 表面图")
	parser.add_argument("--beta_plot_max", type=float, default=None, help="3D 表面图中 beta 的上界；默认使用可行曲线 max(beta)")
	parser.add_argument("--surface_grid", type=int, default=60, help="3D 表面图网格分辨率")
	args = parser.parse_args()

	res = explore_alpha_beta(
		eta=args.eta,
		p=args.p,
		mu_l=args.mu_l,
		mu_r=args.mu_r,
		u_prime=args.u_prime,
		alpha_min=args.alpha_min,
		alpha_max=args.alpha_max,
		num_points=args.num_points,
		beta_min=args.beta_min,
		beta_max=args.beta_max,
		L=args.L,
		h_expr=args.h_expr,
		g_expr=args.g_expr,
		save_dir=args.save_dir,
	)

	print("=" * 80)
	print("u' =", res["u_prime"])
	print("alpha* =", f"{res['alpha_star']:.6g}")
	print("beta* =", f"{res['beta_star']:.6g}")
	print("Obj* =", f"{res['obj_star']:.6g}")
	print("beta_max_eff =", f"{res['effective_beta_max']:.6g}")

	if args.animate:
		gif_path = os.path.join(args.save_dir, args.gif_name)
		animate_alpha_beta_curve(res["alphas"], res["betas"], res["feas"], res["objs"], gif_path)

	if args.plot3d:
		p3d_path = os.path.join(args.save_dir, args.plot3d_name)
		plot_3d_curve(res["alphas"], res["betas"], res["objs"], res["feas"], int(res["best_idx"]), p3d_path)

	if args.plot_surface:
		beta_plot_max = args.beta_plot_max
		if beta_plot_max is None:
			# 以可行曲线 beta 的最大值为上界（略加裕量）
			curve_betas = [b for b, ok in zip(res["betas"], res["feas"]) if ok]
			beta_plot_max = (max(curve_betas) if len(curve_betas) > 0 else res["effective_beta_max"]) * 1.05
		surf_path = os.path.join(args.save_dir, "objective_surface.png")
		plot_objective_surface(
			eta=args.eta,
			p=args.p,
			mu_l=args.mu_l,
			h_fn=res["h_fn"],
			g_fn=res["g_fn"],
			alpha_min=args.alpha_min,
			alpha_max=args.alpha_max,
			beta_min_eff=args.beta_min,
			beta_plot_max=float(beta_plot_max),
			grid_n=args.surface_grid,
			curve_alphas=res["alphas"],
			curve_betas=res["betas"],
			curve_objs=res["objs"],
			curve_feas=res["feas"],
			best_idx=int(res["best_idx"]),
			save_path=surf_path,
		)


if __name__ == "__main__":
	main()
