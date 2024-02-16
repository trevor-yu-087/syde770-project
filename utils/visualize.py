import numpy as np
import matplotlib.pyplot as plt


def pred_vs_error(
        preds, # (Element, Channel) array
        targets, # (Element, Channel) array
        title: None
):
    x = np.arange(preds.shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(20,10), tight_layout=True)

    fig.suptitle(title)

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
