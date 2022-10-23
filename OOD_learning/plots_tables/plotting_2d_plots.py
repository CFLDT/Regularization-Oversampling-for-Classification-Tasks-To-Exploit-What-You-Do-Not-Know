import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from itertools import combinations


def twod_plots_plotter(name, y_train, X, datapipeline,
                       orig_space=False, training_space=False, directions=False, max_combos=None):
    if orig_space == True:
        X = pd.DataFrame(X, columns=datapipeline.colnames)

    if (training_space == True) or (directions == True):
        X = pd.DataFrame(X)

    combos = list(combinations(X.columns, 2))

    base_path = Path(__file__).parent

    i = 0

    for combo in combos:

        if max_combos != None:
            if max_combos < i:
                break
            i = i + 1

        plot_name_2d = "2d_" + name + str(combo) + ".png"

        f, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[combo[0]], X[combo[1]], c=y_train, s=50,
                   cmap="coolwarm_r",
                   edgecolor="white", linewidth=1)

        ax.set(xlabel=combo[0], ylabel=combo[1])

        if training_space == True:
            plt.savefig((base_path / "../../plots/plots 2d plots/training_space" / plot_name_2d).resolve())

        if directions == True:
            plt.savefig((base_path / "../../plots/plots 2d plots/directions" / plot_name_2d).resolve())

        plt.close()
