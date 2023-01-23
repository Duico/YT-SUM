# -*- coding: utf-8 -*-
from os import listdir
from pathlib import Path
import json
import numpy as np
import h5py
from evaluation_metrics import evaluate_summary
from generate_summary import generate_summary
from generate_video import generate_video
import argparse

# arguments to run the script
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,
                    default='../PGL-SUM/Summaries/PGL-SUM/exp1/SumMe/results/split0',
                    help="Path to the json files with the scores of the frames for each epoch")
parser.add_argument("--video_path", type=str,
                    help="Path to the mp4 videos to generate video summaries from, e.g. ../SumMe/videos/", required=True)
parser.add_argument("--dataset", type=str, default='SumMe', help="Dataset to be used")
parser.add_argument("--eval", type=str, default="max", help="Eval method to be used for f_score reduction (max or avg)")
parser.add_argument("--epoch", type=int, default="150", help="Best epoch to be used for evaluation")

args = vars(parser.parse_args())
path = args["path"]
videos_path = args["video_path"]
dataset = args["dataset"]
eval_method = args["eval"]
epoch = str(args["epoch"])

# results = [f for f in listdir(path) if f.endswith(".json")]
# results.sort(key=lambda video: int(video[6:-5]))
dataset_path = '../PGL-SUM/data/datasets/' + dataset + '/eccv16_dataset_' + dataset.lower() + '_google_pool5.h5'
epoch_json = dataset + '_' + epoch + '.json'

# originally: for each epoch
all_scores = []
with open(path + '/' + epoch_json) as f:     # read the json file ...
    data = json.loads(f.read())
    keys = list(data.keys())

    for video_name in keys:             # for each video inside that json file ...
        scores = np.asarray(data[video_name])  # read the importance scores from frames
        all_scores.append(scores)

all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
all_video_path = []

with h5py.File(dataset_path, 'r') as hdf:
    for video_name in keys:
        video_index = video_name[6:]

        user_summary = np.array(hdf.get('video_' + video_index + '/user_summary'))
        sb = np.array(hdf.get('video_' + video_index + '/change_points'))
        n_frames = np.array(hdf.get('video_' + video_index + '/n_frames'))
        positions = np.array(hdf.get('video_' + video_index + '/picks'))
        video_name_original = np.array(hdf.get('video_' + video_index + '/video_name'))

        # only works for SumMe
        video_name_original = np.array2string(video_name_original)[2:-1]
        video_path = Path(videos_path) / Path(video_name_original).with_suffix('.mp4')
        
        all_video_path.append(video_path)
        all_user_summary.append(user_summary)
        all_shot_bound.append(sb)
        all_nframes.append(n_frames)
        all_positions.append(positions)

all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)

all_f_scores = []
# compare the resulting summary with the ground truth one, for each video
for video_index in range(len(all_summaries)):
    summary = all_summaries[video_index]
    user_summary = all_user_summary[video_index]
    f_score = evaluate_summary(summary, user_summary, eval_method)
    video_path = all_video_path[video_index]
    video_output_path = Path(path) / 'video_out' / f"epoch_{epoch}" / video_path.name
    generate_video(video_path, video_output_path, summary)
    print(f_score)
    print(video_path)
    print(user_summary.shape)
    all_f_scores.append(f_score)

# f_score_epochs.append(np.mean(all_f_scores))
print("f_score: ", np.mean(all_f_scores))

# Save the importance scores in txt format.
# with open(path + '/f_scores.txt', 'w') as outfile:
#     json.dump(f_score_epochs, outfile)
