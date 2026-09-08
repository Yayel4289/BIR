import pandas as pd

from exps.skinCancer.data_loader import get_X_y_tensors_from_df
import src.visualization.plots as plots

df = pd.read_csv("exps/skinCancer/data/hmnist_age_label.csv")

X, y = get_X_y_tensors_from_df(df)

plots.plot_image(X, 0)
plots.show()

