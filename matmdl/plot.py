"""
Plots several figures per optimization:

1. Stress-strain curves of all parameter sets (per orientation, if applicable)
2. Stress-strain curve of best parameter set (per orientation, if applicable)
3. Convergence (lowest error as a function of iteration)
4. Histograms of parameter evaluations
5. Partial dependencies of the objective function

Prints best parameters to file out_best_params.txt

TODO: refactor overlap between main() and plot_single()
"""

import os

import matplotlib
import numpy as np
from scipy.optimize import curve_fit
from skopt.plots import plot_evaluations, plot_objective

from .core import optimizer as optimizer
from .core.parallel import Checkout
from .core.parser import uset
from .core.utilities import msg, warn

matplotlib.use("Agg")  # backend selected for cluster compatibility
import matplotlib.pyplot as plt  # noqa: E402

# use local path for plots
with uset.unlock():
	uset.main_path = os.getcwd()


@Checkout("out", local=True)
def run_fast_plots():
	"""Plots all available plot types"""
	msg("\n# start plotting")
	global in_opt
	in_opt = optimizer.InOpt(uset.orientations, uset.params)
	orients = in_opt.orients
	fig0, ax0 = plt.subplots()
	labels0 = []
	colors0 = plt.rcParams["axes.prop_cycle"].by_key()["color"]
	for ct_orient, orient in enumerate(orients):
		data = np.load(os.path.join(os.getcwd(), f"out_time_disp_force_{orient}.npy"))
		num_iter = len(data[0, 0, :])
		# -----------------------------------------------------------------------------------------------
		# plot all trials, in order:
		msg(f"{orient}: all curves")
		fig, ax = plt.subplots()
		for i in range(num_iter):
			eng_strain = data[:, 1, i] / uset.length
			eng_stress = data[:, 2, i] / uset.area
			ax.plot(
				eng_strain,
				eng_stress,
				alpha=0.2 + (i + 1) / num_iter * 0.8,
				color="#696969",
			)

		# plot experimental results:
		exp_filename = uset.orientations[orient]["exp"]
		exp_SS = np.loadtxt(os.path.join(os.getcwd(), exp_filename), skiprows=1, delimiter=",")
		ax.plot(
			exp_SS[:, 0],
			exp_SS[:, 1],
			"-s",
			markerfacecolor="black",
			color="black",
			label="Experimental " + uset.grain_size_name,
		)
		ax0.plot(
			exp_SS[:, 0],
			exp_SS[:, 1],
			"s",
			markerfacecolor=colors0[ct_orient],
			color="black",
			label="Experimental " + uset.grain_size_name,
		)
		labels0.append(f"Exp. [{orient}]")

		# plot best guess:
		errors = np.loadtxt(os.path.join(os.getcwd(), "out_errors.txt"), skiprows=1, delimiter=",")[
			:, -1
		]
		loc_min_error = np.argmin(errors)
		eng_strain_best = data[:, 1, loc_min_error] / uset.length
		eng_stress_best = data[:, 2, loc_min_error] / uset.area
		ax.plot(
			eng_strain_best,
			eng_stress_best,
			"-o",
			alpha=1.0,
			color="blue",
			label="Best parameter set",
		)
		ax0.plot(
			eng_strain_best,
			eng_stress_best,
			"-",
			alpha=1.0,
			linewidth=2,
			color=colors0[ct_orient],
			label="Best parameter set",
		)
		labels0.append(f"Fit [{orient}]")

		# save best stress-strain data:
		temp_ss = np.stack((eng_strain_best, eng_stress_best), axis=1)
		np.savetxt(f"out_best_stressStrain_{orient}.csv", temp_ss, delimiter=",", header="Eng. Strain, Eng. Stress")

		plot_settings(ax)
		if uset.max_strain > 0:
			ax.set_xlim(right=uset.max_strain)
		fig.savefig(
			os.path.join(os.getcwd(), "res_opt_" + orient + ".png"),
			bbox_inches="tight",
			dpi=400,
		)
		plt.close(fig)
		# -----------------------------------------------------------------------------------------------
		# print best paramters
		params = np.loadtxt(
			os.path.join(os.getcwd(), "out_progress.txt"), skiprows=1, delimiter=","
		)[:, 1:]
		# ^ full list: time, then one param per column
		best_params = params[loc_min_error, :]
		with open("out_best_params.txt", "w") as f:
			f.write("Total iterations: " + str(num_iter))
			f.write("\nBest iteration:   " + str(int(loc_min_error)))
			f.write("\nLowest error:     " + str(errors[loc_min_error]) + "\n\n")
			f.write("Best parameters:\n")  # + ", ".join([str(f) for f in best_params]) + "\n\n")
			for name, value in zip(in_opt.params, best_params):
				f.write(f"{name:<10}: {value:>0.2E}\n")
			if len(uset.param_additional_legend) > 0:
				f.write("\nFixed parameters:\n" + ", ".join(uset.param_additional_legend) + "\n")
				f.write(
					"Fixed parameter values:\n"
					+ ", ".join([str(get_param_value(f)) for f in uset.param_additional_legend])
					+ "\n\n"
				)
			f.write("\n")
		# -----------------------------------------------------------------------------------------------
		# plot best paramters
		legend_info = []
		for i, param in enumerate(in_opt.params):
			legend_info.append(f"{name_to_sym(param)}={best_params[i]:.1f}")
		# also add additional parameters to legend:
		for param_name in uset.param_additional_legend:
			legend_info.append(name_to_sym(param_name) + "=" + str(get_param_value(param_name)))
		# add error value
		legend_info.append("Error: " + str(errors[loc_min_error]))
		legend_info = "\n".join(legend_info)

		msg(f"{orient}: best fit")
		fig, ax = plt.subplots()
		ax.plot(
			exp_SS[:, 0],
			exp_SS[:, 1],
			"-s",
			markerfacecolor="black",
			color="black",
			label="Experimental " + uset.grain_size_name,
		)
		ax.plot(
			eng_strain_best,
			eng_stress_best,
			"-o",
			alpha=1.0,
			color="blue",
			label=legend_info,
		)
		plot_settings(ax)
		if uset.max_strain > 0:
			ax.set_xlim(right=uset.max_strain)
		fig.savefig(
			os.path.join(os.getcwd(), "res_single_" + orient + ".png"),
			bbox_inches="tight",
			dpi=400,
		)
		plt.close(fig)

	# finish fig0, the plot of all sims and experimental data
	if len(orients) > 1:
		msg("all stress-strain")
		plot_settings(ax0, legend=False)
		ax0.legend(loc="upper left", bbox_to_anchor=(1.0, 1.02), labels=labels0, fancybox=False)
		if uset.max_strain > 0:
			ax0.set_xlim(right=uset.max_strain)
		fig0.savefig(os.path.join(os.getcwd(), "res_all.png"), bbox_inches="tight", dpi=400)
	else:
		plt.close(fig0)

	# -----------------------------------------------------------------------------------------------
	# plot convergence
	msg("convergence information")
	fig, ax = plt.subplots()
	running_min = np.empty((num_iter))
	running_min[0] = errors[0]
	for i in range(1, num_iter):
		if errors[i] < running_min[i - 1]:
			running_min[i] = errors[i]
		else:
			running_min[i] = running_min[i - 1]
	ax.plot(list(range(num_iter)), running_min, "-o", color="blue")
	plot_settings(ax, legend=False)
	ax.set_xlabel("Iteration number")
	ax.set_ylabel("Lowest RMSE")
	fig.savefig("res_convergence.png", dpi=400, bbox_inches="tight")
	plt.close()
	# -----------------------------------------------------------------------------------------------
	all_errors = np.loadtxt(os.path.join(os.getcwd(), "out_errors.txt"), skiprows=1, delimiter=",")
	plot_error_front(errors=all_errors, samples=in_opt.orients)
	plot_error_front_fit(errors=all_errors, samples=in_opt.orients)
	msg("Finished fast plots")


def run_slow_plots():
	"""
	These require retraining and sampling the surrogate model, and can be quite slow.

	They are separated so that the time spent within the Checkout context can be shortened,
	allowing easier plotting while simulations are still running.
	"""
	msg("Starting slow plots")
	in_opt = optimizer.InOpt(uset.orientations, uset.params)
	opt = optimizer.instantiate(in_opt, uset)
	with Checkout("out", local=True):
		msg("retraining surrogate model")
		opt = optimizer.load_previous(opt, search_local=True)
		if opt._n_initial_points > 0:
			msg(
				f"warning, found only {opt.n_initial_points_ - opt._n_initial_points} points; training on random points..."
			)
			opt._n_initial_points = 0
			fake_x = opt.Xi[-1]
			fake_y = opt.yi[-1]
			opt.Xi = opt.Xi[:-1]
			opt.yi = opt.yi[:-1]
			opt.tell(fake_x, fake_y)
	# plot parameter distribution
	msg("parameter evaluations")
	apply_param_labels(plot_evaluations(opt.get_result()), diag_label="Freq.")
	plt.savefig(fname="res_evaluations.png", bbox_inches="tight", dpi=600, transparent=False)
	plt.close()
	# plot partial dependence
	msg("partial dependencies")
	apply_param_labels(plot_objective(opt.get_result()), diag_label="Objective")
	plt.savefig(fname="res_objective.png", bbox_inches="tight", dpi=600, transparent=False)
	plt.close()

	msg("# stop plotting\n")


@Checkout("out", local=True)
def plot_single():
	"""Plot results of single parameter run"""
	msg("\n# start plotting single")
	fig0, ax0 = plt.subplots()
	in_opt = optimizer.InOpt(uset.orientations, uset.params)
	orients = in_opt.orients
	labels0 = []
	colors0 = plt.rcParams["axes.prop_cycle"].by_key()["color"]
	for ct_orient, orient in enumerate(orients):
		fig, ax = plt.subplots()
		msg(f"plotting {orient}")

		# experimental:
		exp_filename = uset.orientations[orient]["exp"]
		exp_SS = np.loadtxt(os.path.join(os.getcwd(), exp_filename), skiprows=1, delimiter=",")
		ax.plot(
			exp_SS[:, 0],
			exp_SS[:, 1],
			"-s",
			markerfacecolor="black",
			color="black",
			label="Experimental " + uset.grain_size_name,
		)
		ax0.plot(
			exp_SS[:, 0],
			exp_SS[:, 1],
			"s",
			markerfacecolor=colors0[ct_orient],
			color="black",
			label="Experimental " + uset.grain_size_name,
		)
		labels0.append(f"Exp. [{orient}]")

		# simulation:
		data = np.loadtxt(f"temp_time_disp_force_{orient}.csv", delimiter=",", skiprows=1)
		eng_strain = data[:, 1] / uset.length
		eng_stress = data[:, 2] / uset.area
		ax.plot(
			eng_strain,
			eng_stress,
			"-o",
			alpha=1.0,
			color="blue",
			label="Best parameter set",
		)
		ax0.plot(
			eng_strain,
			eng_stress,
			"-",
			alpha=1.0,
			linewidth=2,
			color=colors0[ct_orient],
			label="Best parameter set",
		)
		labels0.append(f"Fit [{orient}]")

		plot_settings(ax)
		if uset.max_strain > 0:
			ax.set_xlim(right=uset.max_strain)
		fig.savefig(
			os.path.join(os.getcwd(), "res_single_" + orient + ".png"),
			bbox_inches="tight",
			dpi=400,
		)
		plt.close(fig)

	# finish fig0, the plot of all sims and experimental data
	if len(orients) > 1:
		msg("all stress-strain")
		plot_settings(ax0, legend=False)
		ax0.legend(loc="best", labels=labels0, fancybox=False, bbox_to_anchor=(1.02, 1))
		if uset.max_strain > 0:
			ax0.set_xlim(right=uset.max_strain)
		fig0.savefig(os.path.join(os.getcwd(), "res_all.png"), bbox_inches="tight", dpi=400)
	else:
		plt.close(fig0)

	msg("# stop plotting single\n")


def get_rotation_ccw(degrees):
	"""Takes data, rotates ccw"""
	radians = degrees / 180.0 * np.pi
	rot = np.array([[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]])
	return rot


def plot_error_front_fit(errors, samples):
	"""
	Plots Pareto efficient pairwise errors with parabolic fits.

	Args:
	    errors: matrix of error values (cols iterations, rows samples)
	    samples: names of each sample
	"""
	num_samples = np.shape(errors)[1] - 1
	if num_samples < 2:
		warn("skipping multi-error plot")
		return
	else:
		msg("error front fits")

	xsize = 2.5
	ysize = 2
	fig, ax = plt.subplots(
		nrows=num_samples - 1,
		ncols=num_samples - 1,
		squeeze=False,
		figsize=(xsize * (num_samples - 1), ysize * (num_samples - 1)),
		layout="constrained",
	)

	rotation = get_rotation_ccw(degrees=45)
	curvatures = {sample: 0.0 for sample in samples}
	diff_xs = {
		sample: 0.0 for sample in samples
	}  # x-shift between parabola center and y=x (equal error)
	diff_ys = {sample: 0.0 for sample in samples}  # y-shift between parabola center and global min
	diff_rs = {
		sample: 0.0 for sample in samples
	}  # height of parabola from line where equal error is 0
	global_best_ind = np.argmin(errors[:, -1])

	for i in range(0, num_samples - 1):  # i horizontal going right
		for j in range(0, num_samples - 1):  # j vertical going down
			_ax = ax[j, i]
			if i > j:
				_ax.axis("off")
			else:
				plt_errors = np.stack((errors[:, i], errors[:, j + 1]), axis=1)

				is_boundary = np.full((np.shape(plt_errors)[0]), False, dtype=bool)
				for k, error in enumerate(plt_errors):
					is_boundary[k] = np.invert(
						np.any(
							np.all(
								np.stack(
									(
										plt_errors[:, 0] < error[0],
										plt_errors[:, 1] < error[1],
									),
									axis=1,
								),
								axis=1,
							),
							axis=0,
						)
					)
				boundary_errors = plt_errors[is_boundary, :]

				_ax.plot(
					boundary_errors[:, 0],
					boundary_errors[:, 1],
					"o",
					color="blue",
					markerfacecolor="none",
					zorder=2.0,
				)
				max_boundary_error = max(max(boundary_errors[:, 0]), max(boundary_errors[:, 1]))
				min_boundary_error = min(min(boundary_errors[:, 0]), min(boundary_errors[:, 1]))
				span = max_boundary_error - min_boundary_error
				pad = 0.05  # fraction of data span to add to window span
				_ax.set_xlim(
					left=min_boundary_error - pad * span,
					right=max_boundary_error + pad * span,
				)
				_ax.set_ylim(
					bottom=min_boundary_error - pad * span,
					top=max_boundary_error + pad * span,
				)
				_ax.plot(
					plt_errors[:, 0],
					plt_errors[:, 1],
					"o",
					color="black",
					markerfacecolor="none",
					zorder=1.0,
				)
				_ax.set_xlabel(f"{samples[i]}")
				_ax.set_ylabel(f"{samples[j+1]}")

				# add equal error line
				line = np.linspace(min_boundary_error, max_boundary_error, 100)
				_ax.plot(line, line, ":", color="grey", zorder=2.5)

				# check if sufficient points in front
				if np.shape(boundary_errors)[0] < 3:
					warn(
						f"Warning: insufficient front found for samples {samples[i]} and {samples[j+1]}",
						RuntimeWarning,
					)
					continue

				# fit with parabola in rotated frame
				fit_data = boundary_errors @ rotation
				# max point of each error becomes (x,y) pair in rotated frame
				minmax_rot = (
					np.asarray(
						[
							boundary_errors[np.argmax(boundary_errors[:, 1])],
							boundary_errors[np.argmax(boundary_errors[:, 0])],
						]
					)
					@ rotation
				)

				# only want x bounds in new frame for curve fitting
				x_rot = np.linspace(minmax_rot[0, 0], minmax_rot[1, 0], 200)

				# also want location of least error in rotated frame
				global_best_loc = plt_errors[global_best_ind] @ rotation

				def f(x, b, h, k):
					return b * (x - h) ** 2 + k

				try:
					sigma = []
					for ii in range(len(fit_data[:, 0])):
						sum_j = 0.0
						for jj in range(len(fit_data[:, 0])):
							if jj == ii:
								continue
							dist = np.abs(fit_data[jj, 0] - fit_data[ii, 0])
							if dist != 0:
								sum_j += dist
						sigma.append(np.abs(fit_data[ii, 0]) / sum_j)
					sigma = np.asarray(sigma)

					popt, _ = curve_fit(
						f,
						fit_data[:, 0],
						fit_data[:, 1],
						p0=(0, 10, 100),
						bounds=((-10, -100, -500), (10, 100, 500)),
						sigma=sigma,
					)
					y_rot = f(x_rot, *popt)
					curve_reg = np.stack((x_rot, y_rot), axis=1) @ rotation.T
					_ax.plot(
						curve_reg[:, 0],
						curve_reg[:, 1],
						"--",
						color="red",
						label="fit",
						zorder=3.0,
					)
					curvatures[samples[i]] = curvatures[samples[i]] + popt[0]
					curvatures[samples[j + 1]] = curvatures[samples[j + 1]] + popt[0]

					# also want difference from global best error:
					diff_ys[samples[i]] = diff_ys[samples[i]] + global_best_loc[1] + popt[2]
					diff_ys[samples[j + 1]] = diff_ys[samples[j + 1]] + global_best_loc[1] + popt[2]
					# and x-shift from equal error (let positive favor lower error for that sample):
					diff_xs[samples[i]] = diff_xs[samples[i]] + popt[1]
					diff_xs[samples[j + 1]] = diff_xs[samples[j + 1]] - popt[1]
					# overall height of parabola in rotated frame:
					diff_rs[samples[i]] = diff_rs[samples[i]] + f(0, *popt)
					diff_rs[samples[j + 1]] = diff_rs[samples[j + 1]] + f(0, *popt)
				except RuntimeError:
					warn(
						f"Warning: unable to fit Pareto front for samples {samples[i]} and {samples[j+1]}",
						RuntimeWarning,
					)

				if i > 0:
					_ax.set_ylabel("")
				if j < num_samples - 2:
					_ax.set_xlabel("")

	with open("out_best_params.txt", "a+") as f:
		# curvatures:
		f.write("Mean pairwise error curvatures:\n")
		for sample in samples:
			f.write(f"    {sample}: {curvatures[sample]/num_samples}\n")
		f.write(f"Mean overall pairwise error curvature:\n{np.mean(list(curvatures.values()))}\n\n")

		# x-shifts between global min and parabola center:
		f.write("Mean pairwise x-shifts:\n")
		for sample in samples:
			f.write(f"    {sample}: {diff_xs[sample]/num_samples}\n")
		f.write(f"Mean overall pairwise x-shifts:\n{np.mean(list(diff_xs.values()))}\n\n")

		# y-shifts between global min and parabola center:
		f.write("Mean pairwise y-shifts:\n")
		for sample in samples:
			f.write(f"    {sample}: {diff_ys[sample]/num_samples}\n")
		f.write(f"Mean overall pairwise y-shifts:\n{np.mean(list(diff_ys.values()))}\n\n")

		# distance from parabola to line of y=-x (so no rotated x-shift included):
		f.write("Mean pairwise heights above 0 error:\n")
		for sample in samples:
			f.write(f"    {sample}: {diff_rs[sample]/num_samples}\n")
		f.write(f"Mean overall pairwise heights:\n{np.mean(list(diff_rs.values()))}\n\n")

	fig.savefig(os.path.join(os.getcwd(), "res_errors_fit.png"), bbox_inches="tight", dpi=600)
	plt.close(fig)


def plot_error_front(errors, samples):
	"""plot Pareto frontiers of error from each pair of samples

	TODO:
		- focus on minimal front?
		- check for convexity of each pairwise cases? e.g. area between hull and front
	"""
	num_samples = np.shape(errors)[1] - 1
	if num_samples < 2:
		warn("skipping multi-error plot")
		return
	else:
		msg("error fronts")

	size = 2  # size in inches of each suplot here
	fig, ax = plt.subplots(
		nrows=num_samples - 1,
		ncols=num_samples - 1,
		squeeze=False,
		figsize=(1.6 * size, size)
		if num_samples == 2
		else (size * (num_samples - 1), size * (num_samples - 1)),
		layout="constrained",
	)

	ind_min_error = np.argmin(errors[:, -1])
	for i in range(0, num_samples - 1):  # i horizontal going right
		for j in range(0, num_samples - 1):  # j vertical going down
			_ax = ax[j, i]
			if i > j:
				_ax.axis("off")
			else:
				_ax.scatter(errors[:, i], errors[:, j + 1], c=errors[:, -1], cmap="viridis")
				_ax.set_xlabel(f"{samples[i]} Error")
				_ax.set_ylabel(f"{samples[j+1]} Error")

				if i > 0:
					_ax.set_yticklabels([])
					_ax.set_ylabel("")
				if j < num_samples - 2:
					_ax.set_xticklabels([])
					_ax.set_xlabel("")

				# global min:
				_ax.plot(
					errors[ind_min_error, i],
					errors[ind_min_error, j + 1],
					"*",
					color="red",
					markersize=12,
				)

	fig.colorbar(
		matplotlib.cm.ScalarMappable(
			norm=matplotlib.colors.Normalize(
				vmin=min(errors[:, -1]),
				vmax=max(errors[:, -1]),
			),
			cmap="viridis",
		),
		ax=ax[0, 0] if num_samples == 2 else ax[0, 1],
		label="Mean Error",
		pad=0.05 if num_samples == 2 else -1,
		aspect=15,
	)
	fig.savefig(os.path.join(os.getcwd(), "res_errors.png"), bbox_inches="tight", dpi=400)
	plt.close(fig)


def plot_settings(ax, legend=True):
	ax.set_xlabel("Engineering Strain, m/m")
	ax.set_ylabel("Engineering Stress, MPa")
	ax.set_xlim(left=0)
	ax.set_ylim(bottom=0)
	if legend:
		ax.legend(loc="best", fancybox=False)
	plt.tick_params(which="both", direction="in", top=True, right=True)
	ax.set_title(uset.title)


def apply_param_labels(ax_array, diag_label):
	shape = np.shape(ax_array)
	if len(shape) == 0:
		ax_array.set_xlabel(name_to_sym(in_opt.params[0]))
		return
	for i in range(shape[0]):
		for j in range(shape[1]):
			ax_array[i, j].set_ylabel(name_to_sym(in_opt.params[i]))
			ax_array[i, j].set_xlabel(name_to_sym(in_opt.params[j]))

			if i == j:  # diagonal subplots
				plt.setp(
					ax_array[i, j].get_xticklabels(),
					rotation=45,
					ha="left",
					rotation_mode="anchor",
				)
				ax_array[i, j].set_ylabel(diag_label)
				ax_array[i, j].tick_params(axis="y", labelleft=False, labelright=True, left=False)

			if (j > 0) and not (i == j) and not (i == shape[0] - 1):  # middle section
				ax_array[i, j].tick_params(axis="y", left=False)
				ax_array[i, j].set_ylabel(None)
				ax_array[i, j].set_xlabel(None)

			if (i < shape[0] - 1) and not (i == j):  # not bottom row
				ax_array[i, j].tick_params(axis="x", bottom=False)

			if (i == shape[0] - 1) and not (i == j):  # bottom row
				plt.setp(
					ax_array[i, j].get_xticklabels(),
					rotation=45,
					ha="right",
					rotation_mode="anchor",
				)

			if (i == shape[0] - 1) and not (i == j) and not (j == 0):  # middle bottom row
				ax_array[i, j].tick_params(axis="y", left=False)
				ax_array[i, j].set_ylabel(None)

	return ax_array


def name_to_sym(name, cap_sense=False):
	name_to_sym_dict = {
		"Tau0": r"$\tau_0$",
		"Tau01": r"$\tau_0^{(1)}$",
		"Tau02": r"$\tau_0^{(2)}$",
		"H0": r"$h_0$",
		"H01": r"$h_0^{(1)}$",
		"H02": r"$h_0^{(2)}$",
		"TauS": r"$\tau_s$",
		"TauS1": r"$\tau_s^{(1)}$",
		"TauS2": r"$\tau_s^{(2)}$",
		"q": r"$q$",
		"q1": r"$q_1$",
		"q2": r"$q_2$",
		"hs": r"$h_s$",
		"hs1": r"$h_s^{(1)}$",
		"hs2": r"$h_s^{(2)}$",
		"gamma0": r"$\gamma_0$",
		"gamma01": r"$\gamma_0^{(1)}$",
		"gamma02": r"$\gamma_0^{(2)}$",
		"g0": r"$\gamma_0$",
		"f0": r"$f_0$",
		"f1": r"$f_1$",
		"f2": r"$f_2$",
		"f3": r"$f_3$",
		"f4": r"$f_4$",
		"f5": r"$f_5$",
		"f01": r"$f_0^{(1)}$",
		"f02": r"$f_0^{(2)}$",
		"q0": r"$q_0$",
		"qA1": r"$q_{A1}$",
		"qB1": r"$q_{B1}$",
		"qA2": r"$q_{A2}$",
		"qB2": r"$q_{B2}$",
		"g0exp": r"log($\gamma_0$)",
	}
	if cap_sense is True:
		have_key = name in name_to_sym_dict.keys()
	else:
		have_key = name.lower() in [key.lower() for key in name_to_sym_dict.keys()]
		name_to_sym_dict = {key.lower(): value for key, value in name_to_sym_dict.items()}
		name = name.lower()

	if have_key:
		return name_to_sym_dict[name]
	elif "_deg" in name:
		return name[:-4] + " rot."
	elif "_mag" in name:
		return name[:-4] + " mag."
	else:
		warn(f"Unknown parameter name: {name}", UserWarning)
		return str(name)


def get_param_value(param_name):
	with open(uset.param_file, "r") as f1:
		lines = f1.readlines()
	for line in lines:
		if line[: line.find("=")].strip() == param_name:
			return line[line.find("=") + 1 :].strip()


if __name__ == "__main__":
	if uset.do_single:
		plot_single()
	else:
		run_fast_plots()
		run_slow_plots()
