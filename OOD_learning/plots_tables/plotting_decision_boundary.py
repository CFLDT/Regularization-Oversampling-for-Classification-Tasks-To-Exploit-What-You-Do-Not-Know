import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def dec_boundary_plotter(name, method, X, y, xx,
                         yy, grid_probs_fr, ood_train, grid_probs_ood, ood_samp):

    base_path = Path(__file__).parent
    plot_name_prob = "prob_boundary_" + name + "_" + method + ".png"
    plot_path_prob = (base_path / "../../plots/plots decision boundaries" / plot_name_prob).resolve()

    plot_name_prob_2 = "prob_boundary_with_unlab_" + name + "_" + method + ".png"
    plot_path_prob_2 = (base_path / "../../plots/plots decision boundaries" / plot_name_prob_2).resolve()

    probs_fr = grid_probs_fr.reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs_fr, 25, cmap="coolwarm_r",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X[:, 0][(ood_train == 0)],
               X[:, 1][(ood_train == 0)], c=y[(ood_train == 0)], s=50,
               cmap="coolwarm_r", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(np.min(xx), np.max(xx)), ylim=(np.min(yy), np.max(yy)),
           xlabel="$X_1$", ylabel="$X_2$")

    plt.savefig(plot_path_prob)
    plt.close()

    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs_fr, 25, cmap="coolwarm_r",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X[:, 0], X[:, 1], c=y, s=50,
               cmap="coolwarm_r",
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(np.min(xx), np.max(xx)), ylim=(np.min(yy), np.max(yy)),
           xlabel="$X_1$", ylabel="$X_2$")

    plt.savefig(plot_path_prob_2)
    plt.close()

    # ood

    if ood_samp == True:
        plot_name_prob = "ood_probability_boundary_" + name + "_" + method + ".png"
        plot_path_prob = (base_path / "../../plots/plots decision boundaries" / plot_name_prob).resolve()

        probs_ood = grid_probs_ood.reshape(xx.shape)

        f, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(xx, yy, probs_ood, 25, cmap="coolwarm_r",
                              vmin=0, vmax=1)
        ax_c = f.colorbar(contour)
        ax_c.set_label("$P(ood = 1)$")
        ax_c.set_ticks([0, .25, .5, .75, 1])

        ax.scatter(X[:, 0], X[:, 1], c=y + 2 * ood_train, s=50,
                   cmap="coolwarm_r",
                   edgecolor="white", linewidth=1)

        ax.set(aspect="equal",
               xlim=(np.min(xx), np.max(xx)), ylim=(np.min(yy), np.max(yy)),
               xlabel="$X_1$", ylabel="$X_2$")

        plt.savefig(plot_path_prob)
        plt.close()
