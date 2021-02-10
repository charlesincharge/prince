import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import prince


def main():
    n_shapes = 4
    n_points = 12
    n_dims = 2

    shape_sizes = np.arange(1, n_shapes + 1)
    shape_angle_offsets = 10 * np.arange(n_shapes)
    shape_center_offsets = np.tile(np.arange(n_shapes), (n_dims, 1))

    base_angles = np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)
    # Size (n_shapes, n_points)
    angles = base_angles[np.newaxis, :] + shape_angle_offsets[:, np.newaxis]

    # Calculate along dimensions
    x = (
        np.cos(angles) * shape_sizes[:, np.newaxis]
        + shape_center_offsets[0][:, np.newaxis]
    )
    y = (
        np.sin(angles) * shape_sizes[:, np.newaxis]
        + shape_center_offsets[1][:, np.newaxis]
    )

    shapes = np.stack([x, y], axis=-1)
    shapes_df = pd.concat([pd.DataFrame(s, columns=['x', 'y']) for s in shapes], keys=np.arange(4), names=['shape_idx'])

    gpa = prince.GPA()
    aligned_shapes = gpa.fit_transform(shapes)
    aligned_shapes_df = pd.concat([pd.DataFrame(s, columns=['x', 'y']) for s in aligned_shapes], keys=np.arange(4), names=['shape_idx'])

    # Plot shapes
    fig, axes = plt.subplots(2)
    ax = sns.scatterplot(data=shapes_df, x='x', y='y', hue='shape_idx', style='shape_idx', ax=axes[0])
    ax.set_title('Before GPA')

    ax = sns.scatterplot(data=aligned_shapes_df, x='x', y='y', hue='shape_idx', style='shape_idx', ax=axes[1])
    ax.set_title('After GPA')

    plt.show()


if __name__ == '__main__':
    main()
