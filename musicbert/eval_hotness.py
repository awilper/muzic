# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

from fairseq.models.roberta import RobertaModel
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import sklearn.metrics
import tables
import sys
import os

# label.id file only contains song ids and we need the msd ids.. since we didn't save those just backwards engineering now
# recall one msd maps to several midi ids 


def get_midi_id_to_msd_id():
    # each midi can map to more than one msd... but we can backwards engineer which one via the hotness scores
    def get_id(file_name):
        return file_name.split('/')[-1].split('.')[0]

    def get_file_paths(directory):
        file_paths = []  # List to store full paths of files
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        return file_paths

    # Replace 'lmd_matched_h5' with the correct path to your folder
    directory = 'lmd_aligned'
    all_file_paths = get_file_paths(directory)
    print(len(all_file_paths))

    # Printing all file paths
    midi_id_to_msd_id = defaultdict(list)
    for path in all_file_paths:
        midi_id = get_id(path)
        msd_id = path.split("/")[-2]
        midi_id_to_msd_id[midi_id] = msd_id 
    return midi_id_to_msd_id

midi_id_to_msd_id = get_midi_id_to_msd_id()


RESULTS_PATH = ""

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def msd_id_to_h5(msd_id):
    """Given an MSD ID, return the path to the corresponding h5"""
    return os.path.join(RESULTS_PATH, 'lmd_matched_h5',
                        msd_id_to_dirs(msd_id) + '.h5')

max_length = 8192 if 'disable_cp' not in os.environ else 1024
batch_size = 4
n_folds = 1

scores = dict()
for score in ["f1_score", "roc_auc_score"]:
    for average in ["macro", "micro", "weighted", "samples"]:
        scores[score + "_" + average] = []
        


def label_fn(label, label_dict):
    return label_dict.string(
        [label + label_dict.nspecial]
    )


# NOTE ONLY LOOKING AT FOLD1
for i in range(n_folds):
    file_base = sys.argv[2].replace("x", str(i)) + "/"
    file_path = file_base + "test.id"  # this needs to be manually copied over

    msd_id_vec = []
    midi_id_vec = []
    hotness_vec = []
    title_vec = []

    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and add to the list
            midi_id = line.strip()
            msd_id = midi_id_to_msd_id[midi_id]
            msd_id_vec.append(msd_id)
            midi_id_vec.append(midi_id)

    #genre_map_file = file_base + "midi_genre_map.json"
    #genre_map = json.load(open(genre_map_file, "r"))
    #genre_map_topmagd = genre_map
    print("starting to fetch hotness")
    for msd_id in msd_id_vec:
        with tables.open_file(msd_id_to_h5(msd_id)) as h5:
            # for now just grab the first one that's matched
            hotness = h5.root.metadata.songs.cols.song_hotttnesss[0]
            hotness_vec.append(hotness)
            title = str(h5.root.metadata.songs.cols.title[0])
            title_vec.append(title)
    print("done fetching hotness")
    #print(len(genre_map["topmagd"]))
    #print(msd_vec)
    print('loading model and data')
    print('start evaluating fold {}'.format(i))

    roberta = RobertaModel.from_pretrained(
        '.',
        checkpoint_file=sys.argv[1].replace('x', str(i)),
        data_name_or_path=sys.argv[2].replace('x', str(i)),
        user_dir='musicbert'
    )
    #num_classes = 13 if 'topmagd' in sys.argv[1] else 25
    num_classes = 2
    roberta.task.load_dataset('valid')
    dataset = roberta.task.datasets['valid']
    label_dict = roberta.task.label_dictionary
    pad_index = label_dict.pad()
    roberta.cuda()
    roberta.eval()

    cnt = 0

    y_true = []
    y_pred = []

    def padded(seq):
        pad_length = max_length - seq.shape[0]
        assert pad_length >= 0
        return np.concatenate((seq, np.full((pad_length,), pad_index, dtype=seq.dtype)))
    
    print(len(dataset))
    for i in range(0, len(dataset), batch_size):
        target = np.vstack(tuple(padded(dataset[j]['target'].numpy()) for j in range(
            i, i + batch_size) if j < len(dataset)))
        target = torch.from_numpy(target)
        target = F.one_hot(target.long(), num_classes=(num_classes + 4))
        target = target.sum(dim=1)[:, 4:]
        source = np.vstack(tuple(padded(dataset[j]['source'].numpy()) for j in range(
            i, i + batch_size) if j < len(dataset)))
        source = torch.from_numpy(source)
        output = torch.sigmoid(roberta.predict(
            'topmagd_head' if 'topmagd' in sys.argv[1] else 'hotness_head', source, True))
        y_true.append(target.detach().cpu().numpy())
        y_pred.append(np.round(output.detach().cpu().numpy()))

        #print("ID {} y_true {} y_pred {}".format(msd_vec[i], y_true[i], y_pred[i]))
        print('evaluating: {:.2f}%'.format(
            i / len(dataset) * 100), end='\r', flush=True)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    y_true = ['H' if int(pair[0]) == 0 else 'C' for pair in y_true]
    y_pred = ['H' if int(pair[0]) == 0 else 'C' for pair in y_pred]
    hotness_vec = [hotness if not np.isnan(hotness) else 0 for hotness in hotness_vec]

    for i in range(20):
        print("hotness {} y true {} y pred {} msd id {} midi id {}".format(hotness_vec[i], y_true[i], y_pred[i], msd_id_vec[i], midi_id_vec[i]))
    output_data = {"y_true": y_true, "y_pred": y_pred,  "hotness": hotness_vec, "title": title_vec, "msd_id": msd_id_vec, "midi_id": midi_id_vec}
    json.dump(output_data, open("metrics.json", "w"))

    for i in range(num_classes):
        print(i, label_fn(i, label_dict))

    print(y_true.shape)
    print(y_pred.shape)

    # with open('genre.npy', 'wb') as f:
    #    np.save(f, {'y_true': y_true, 'y_pred': y_pred})

    for score in ["f1_score", "roc_auc_score"]:
        for average in ["macro", "micro", "weighted", "samples"]:
            try:
                y_score = np.round(y_pred) if score == "f1_score" else y_pred
                result = sklearn.metrics.__dict__[score](
                    y_true, y_score, average=average)
                print("{}_{}:".format(score, average), result)
                scores[score + "_" + average].append(result)
            except BaseException as e:
                print("{}_{}:".format(score, average), e)
                scores[score + "_" + average].append(None)


print(scores)
for k in scores:
    print(k, sum(scores[k]) / len(scores[k]))
