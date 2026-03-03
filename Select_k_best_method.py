from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import numpy as np

X = np.load("data_loader/DB/Features.npy")
y = np.load("data_loader/DB/Labels.npy")
k = int(X.shape[1] / 2)
selector = SelectKBest(score_func=f_regression, k=k)

selector.fit(X, y)

selected_indices = selector.get_support(indices=True)

X_selected = X[:, selected_indices]

print(f"Selected feature indices: {selected_indices}")
print(f"Selected features shape: {X_selected.shape}")

feature_scores = selector.scores_
print("Feature scores:", feature_scores)

np.save("data_loader/DB/selected_features.npy", X_selected)


