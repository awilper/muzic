import tables
import os

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def msd_id_to_h5(h5):
    """Given an MSD ID, return the path to the corresponding h5"""
    return os.path.join('lmd_matched_h5',
                        msd_id_to_dirs(msd_id) + '.h5')

seen_msd_id = set() # many songs can be matched

file_name = "hotness_data_raw/0/test.id"
f = open(file_name)
ids = f.read().split("\n")
f.close()

data = []
for msd_id in ids:
    with tables.open_file(msd_id_to_h5(msd_id)) as h5:
        hotness = h5.root.metadata.songs.cols.song_hotttnesss[0]
        artist = h5.root.metadata.songs.cols.artist_name[0]
        title = h5.root.metadata.songs.cols.title[0]
        release = h5.root.metadata.songs.cols.release[0]
        data += (title, artist, release, hotness, msd_id)

print(data)
