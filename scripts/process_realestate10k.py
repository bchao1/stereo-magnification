import os
import random 
import numpy as np
from pathlib import Path
import datetime
from urllib.parse import parse_qs, urlparse
import multiprocessing as mp

random.seed(1)

def get_video_metadata_files(root, mode):

    data_folder = os.path.join(root, mode)
    video_metadata_files = os.listdir(data_folder)
    video_metadata_files.sort()

    print(f"Total {mode} videos:", len(video_metadata_files))
    return video_metadata_files

def make_data_dirs(name):
    os.makedirs(name, exist_ok=True)
    os.makedirs(name + "/train", exist_ok=True)
    os.makedirs(name + "/test", exist_ok=True)
    
def convert_microsecs(microsecs):
    secs = microsecs/1000000
    return datetime.timedelta(seconds=secs)

class VideoDownloader:
    def __init__(self, metadata_root, train_metadata_root, images_root, videos_root, mode):
        self.metadata_root = metadata_root 
        self.train_metadata_root = train_metadata_root
        self.images_root = images_root
        self.videos_root = videos_root
        self.mode = mode
        self.video_ids = {}
        self.camera_metadata_to_video_id = {}
    
    def download_single_video_by_id(self, video_id):
        video_url = f"http://www.youtube.com/watch?v={video_id}"
        print("Downloading video:", video_url)
        video_path = os.path.join(self.videos_root, self.mode, f"{video_id}.mp4")
        cmd = f"youtube-dl -f best -o {video_path} {video_url}" # best quality
        os.system(cmd)
        if os.path.exists(video_path):
            # only make image directory if video is successfully downloaded
            os.makedirs(os.path.join(self.images_root, self.mode, video_id), exist_ok=True)
        else:
            self.video_ids.pop(video_id, None) # remove video id from dict if video not downloaded

    def extract_frames_by_video_id(self, video_id):
        video_path = os.path.join(self.videos_root, self.mode, f"{video_id}.mp4")
        for timestamp in self.video_ids[video_id]:
            frame_name = f"{video_id}_{timestamp}.jpg"
            frame_save_path = os.path.join(images_root, mode, video_id, frame_name)
            # Extracting frame at a specific timestamp
            # https://stackoverflow.com/questions/27568254/how-to-extract-1-screenshot-for-a-video-with-ffmpeg-at-a-given-time/27573049
            cmd = f"ffmpeg -ss '{timestamp}us' -i {video_path} -vframes 1 -q:v 2 {frame_save_path}"
            os.system(cmd)
    
    def copy_training_metadata(self):
        for camera_metadata_id, video_id in self.camera_metadata_to_video_id.items():
            if video_id in self.video_ids:
                os.system(f"cp {self.metadata_root}/{self.mode}/{camera_metadata_id} {self.train_metadata_root}/{self.mode}/")
        
        
    def download_videos_by_ids(self, split_ids):
        for id_ in split_ids:
            video_metadata_file = os.path.join(metadata_root, mode, id_)
            video_url = None
            with open(video_metadata_file) as file:
                for i, line in enumerate(file):
                    line = line.strip()
                    if i == 0: # get video url
                        video_url = line
                        video_id = parse_qs(urlparse(video_url).query).get('v')[0]
                        self.camera_metadata_to_video_id[id_] = video_id
                        if video_id not in self.video_ids:
                            self.video_ids[video_id] = [] # store time stamps
                    else:
                        timestamp = int(line.split(" ")[0]) # in microsecs
                        self.video_ids[video_id].append(timestamp)
                    
        download_pool = mp.Pool(processes = 100)
        res = download_pool.map_async(self.download_single_video_by_id, self.video_ids.keys())
        res.get()
        
        extract_pool = mp.Pool(processes = 100)
        res = extract_pool.map_async(self.extract_frames_by_video_id, self.video_ids.keys())
        res.get()
        
        self.copy_training_metadata()
    

mode = "train"
metadata_root = "/mnt/data/bchao/MPI/RealEstate10K"
images_root = "/mnt/data/bchao/MPI/images"
videos_root = "/mnt/data/bchao/MPI/videos"
train_metadata_root = "/mnt/data/bchao/MPI/camera_metadata"

make_data_dirs(images_root)
make_data_dirs(videos_root)
make_data_dirs(train_metadata_root)


subset_split_id = 0
#test_subset_size = train_subset_size // 10

video_metadata_files_train = get_video_metadata_files(metadata_root, mode)

num_splits = 100
split_ids = np.array_split(video_metadata_files_train, num_splits)[subset_split_id]
print(len(split_ids))
#for id_ in split_ids:
#    os.system(f"cp ./{metadata_root}/{mode}/{id_} camera_metadata/train/")

downloader = VideoDownloader(metadata_root, train_metadata_root, images_root, videos_root, mode)
downloader.download_videos_by_ids(split_ids)

