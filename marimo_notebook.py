

import marimo

__generated_with = "0.13.1-dev1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import sklearn.datasets as skdatasets

    np.random.seed(0)

    dataset = skdatasets.load_breast_cancer()

    features_used = [-3, -8]
    X = dataset.data[:, features_used]
    feature_names = dataset.feature_names[features_used]

    # min-max normalize the features along the columns
    X_min_vals = X.min(axis=0)
    X_max_vals = X.max(axis=0)
    X = (X - X_min_vals) / (X_max_vals - X_min_vals)

    Y = dataset.target
    target_names = dataset.target_names

    sigmoid = lambda x: 1 / (1 + np.exp(-x))


    def bce(y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


    def create_plots():
        fig = plt.figure(figsize=(16 / 9.0 * 4, 4 * 1.25), layout="constrained")
        fig.suptitle("Logistic Regression")

        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Prediction Probabilities")
        ax.view_init(20, -35)

        return ax


    def plot_graphs(
        ax,
        features,
        labels,
        predictions,
        points_x,
        points_y,
        surface_predictions,
    ):
        # Plot Logistic Regression Predictions
        # Ground truth and training data
        ground_truth_legend = ax.scatter(
            features[:, 0],
            features[:, 1],
            labels,
            color="red",
            alpha=0.5,
            label="Ground Truth",
        )
        # Logistic Regression Predictions
        predictions_legend = ax.scatter(
            features[:, 0],
            features[:, 1],
            predictions,
            color="blue",
            alpha=0.2,
            label="Prediction",
        )
        ax.plot_surface(
            points_x,
            points_y,
            surface_predictions.reshape(dims, dims),
            color="blue",
            alpha=0.2,
        )
        ax.legend(
            (ground_truth_legend, predictions_legend),
            ("Ground Truth", "Predictions"),
            loc="upper left",
        )
        plt.show()

    dims = 10
    points = np.linspace(0, 1, dims)
    points_x, points_y = np.meshgrid(points, points)
    surface_points = np.column_stack((points_x.flatten(), points_y.flatten()))

    w1_slider = mo.ui.slider(
        -15, 0, 1, value=-11, show_value=True, label="$w1$"
    )
    w2_slider = mo.ui.slider(
        -15, 0, 1, value=-11, show_value=True, label="$w2$"
    )
    b_slider = mo.ui.slider(
        0, 10, 1, value=8, show_value=True, label="$b$"
    )

    mo.hstack([mo.vstack([w1_slider, w2_slider]), mo.vstack([b_slider])])
    return (
        X,
        Y,
        b_slider,
        create_plots,
        np,
        plot_graphs,
        points_x,
        points_y,
        sigmoid,
        surface_points,
        w1_slider,
        w2_slider,
    )


@app.cell(hide_code=True)
def _(
    X,
    Y,
    b_slider,
    create_plots,
    np,
    plot_graphs,
    points_x,
    points_y,
    sigmoid,
    surface_points,
    w1_slider,
    w2_slider,
):
    weights = np.array([w1_slider.value, w2_slider.value])
    bias = b_slider.value

    ax = create_plots()

    predictions = np.array([])
    surface_predictions = np.array([])

    # fit the model on the training data
    for x, y in zip(X, Y):
        output = sigmoid(np.dot(weights, x) + bias)

        predictions = np.append(predictions, output)


    for surface_point in surface_points:
        output = sigmoid(np.dot(weights, surface_point) + bias)
        surface_predictions = np.append(surface_predictions, output)

    plot_graphs(
        ax,
        X,
        Y,
        predictions,
        points_x,
        points_y,
        surface_predictions,
    )
    return


if __name__ == "__main__":
    app.run()
