# 6.8611 Final Project 

Since our project depends heavily on MusicBERT's utilities, we forked the repo and added our own code as needed. By viewing https://github.com/awilper/muzic/pull/1 one can see what files are ours. 

Due to versioning issues, it was not possible to run MusicBERT on Google Colab. Instead, we ordered a T4 server on GCP and installed installed the required libraries via Anaconda. You can see in requirements.txt that this required pinning a version of fairseq. Note some nvidia-smi setup was also necessary.  


The commands used to generate our data for classification are as follows. 

1. `python -u gen_hotness.py`
2. `train_genre.sh hotness 2 0 checkpoints/checkpoint_last_musicbert_small.pt > output.log`
3. `python -u eval_hotness.py checkpoints/checkpoint_last_genre_hotness_0_checkpoint_last_musicbert_small.pt hotness_data_bin/0`

We used the results of the second command to produce accuracy plots, and the third command to produce certain metric graphs. 
