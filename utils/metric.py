import numpy as np

def compute_ate(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))

def compute_rte(pred, target, delta, max_delta=-1):
    if max_delta == -1:
        max_delta = pred.shape[0]
    deltas = np.array([delta]) if delta > 0 else np.arange(1, min(pred.shape[0], max_delta))
    rtes = np.zeros(deltas.shape[0])
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.

        err = pred[deltas[i]//32:,:] + target[:-(deltas[i]//32),:] - pred[:-(deltas[i]//32),:] - target[deltas[i]//32:,:]
        rtes[i] = np.sqrt(np.mean(err ** 2))

    # The average of RTE of all window sized is returned.
    return np.mean(rtes)

def compute_ate_rte(pred, target, pred_per_min):
    ate = compute_ate(pred, target)
    if pred.shape[0]*32 < pred_per_min:
        ratio = pred_per_min / pred.shape[0]*32
        rte = compute_rte(pred, target, delta=pred.shape[0] - 1) * ratio
    else:
        rte = compute_rte(pred, target, delta=pred_per_min)

    return ate, rte