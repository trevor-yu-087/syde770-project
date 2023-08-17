import numpy as np
import matplotlib.pyplot as plt


def pred_vs_error(
        epoch: int,
        preds: list,
        targets: list
):
    preds = np.vstack(np.array(preds))
    targets = np.vstack(np.array(targets))
    x = np.arange(preds.shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(20,10), tight_layout=True)

    fig.suptitle(f'Epoch {epoch}')

    axes[0].set_title('X')
    axes[0].set_ylabel('Velocity')
    axes[0].set_xlabel('Velocity Frame')
    axes[0].plot(x, preds[:,0])
    axes[0].plot(x, targets[:,0], alpha=0.3)

    axes[1].set_title('Y')
    axes[1].set_ylabel('Velocity')
    axes[1].set_xlabel('Velocity Frame')
    axes[1].plot(x, preds[:,1])
    axes[1].plot(x, targets[:,1], alpha=0.3)

    axes[2].set_title('Z')
    axes[2].set_ylabel('Velocity')
    axes[2].set_xlabel('Velocity Frame')
    axes[2].plot(x, preds[:,2])
    axes[2].plot(x, targets[:,2], alpha=0.3),

    plt.show()
