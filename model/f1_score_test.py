import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from sklearn import metrics
import math

def compute_f1_score(pred: np.ndarray, target: np.ndarray):
    n_shots = 100
    top_percent = 15
    pred_100 = interpolate_pred(pred, n_shots)
    x_space = np.linspace(0, n_shots, pred.shape[0], endpoint=False)
    target_100 = interp1d(x_space, target, kind='nearest', assume_sorted=True)(np.arange(0, 100))
    partition_elem = int(math.floor(top_percent * n_shots / 100))
    pred_100_bool = np.zeros_like(pred_100)
    target_100_bool = np.zeros_like(target_100)
    pred_100_bool[np.argpartition(pred_100, len(pred_100)-partition_elem-1)[-partition_elem:]] = 1
    target_100_bool[np.argpartition(target_100, len(target_100)-partition_elem-1)[-partition_elem:]] = 1

    f1_score = metrics.f1_score(target_100_bool, pred_100_bool)

    # TODO: IN CASE N_SHOTS != 100
    # x_integral = np.linspace(0, duration_ms, 10000)
    # shots_bins = np.arange(0, duration_ms, SHOT_DURATION_MS)
    # x_pdf = heat_markers_spline(x_integral)
    # bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(x_integral, x_pdf, statistic='mean', bins=shots_bins)

    # shots_durations = bin_edges[1:]-bin_edges[:-1]
    # shots_durations = shots_durations.astype(int)
    # shots_selected = knapSack(math.floor(SUMMARY_DURATION_RATIO * duration_ms), shots_durations, bin_means, len(bin_means))
    
    return f1_score


def interpolate_pred(pred, n_shots = 100):
    x_space = np.linspace(0, n_shots, pred.shape[0], endpoint=False)
    pred_interp = interp1d(x_space, pred, kind='nearest', assume_sorted=True) # could use kind='linear' here
    x_integral = np.linspace(0,n_shots-1,num=5000)
    pred_100, pred_100_bin_edges, _ = binned_statistic(x_integral, pred_interp(x_integral), statistic='mean', bins=n_shots) # TODO check if bins range is end-exclusive
    return pred_100