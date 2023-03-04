import json
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from model.f1_score_test import interpolate_pred


parser = argparse.ArgumentParser()
parser.add_argument('json_path', help="path to a .json file with results")
# parser.add_argument('--features_mode', default='rgb')
parser.add_argument('--video_heatmarkers_dir', required=True, help="path to the directory with heat-markers video-id.h5 files")
# parser.add_argument('--out', help="path to output h5 file", default='out.h5')
args = parser.parse_args()


with open(args.json_path, 'r') as f:
    pred = json.load(f)
    for video_id in pred.keys():
        try:
            heat_markers_pd = pd.read_hdf(args.video_heatmarkers_dir+'/'+video_id+'.h5')
        except:
            print(f"Skipping {video_id}, .h5 not found")
            continue
        fig = plt.figure()
        # gs = GridSpec(2, 10, figure=fig)
        gs = GridSpec(2,1, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax1.set_title("Prediction")
        ax1.ticklabel_format(style='plain')
        ax1.xaxis.set_label('Time (s)')
        ax1.xaxis.set_major_formatter(lambda x, pos :  f"{int(x / 1000)}s")
        ax1.yaxis.set_label('Intensity score')
        current_pred = pred[video_id]
        ax1.set_ylim(0.0,1.0)
        ax1.plot(np.linspace(0, len(heat_markers_pd.timeRangeStartMillis)*heat_markers_pd.markerDurationMillis[0], len(current_pred)), current_pred, '-',  color='orange')
        ax1.bar(heat_markers_pd.timeRangeStartMillis, interpolate_pred(np.array(current_pred), 100), width=heat_markers_pd.markerDurationMillis)

        ax2.set_title("Ground truth")
        ax2.ticklabel_format(style='plain')
        ax2.xaxis.set_label('Time (ms)')
        ax2.yaxis.set_label('Intensity score')
        ax2.set_ylim(0.0,1.0)
        ax2.bar(heat_markers_pd.timeRangeStartMillis, heat_markers_pd.heatMarkerIntensityScoreNormalized, width=heat_markers_pd.markerDurationMillis)
        
        plt.show()



