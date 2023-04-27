import cv2
import json
import numpy as np
import os
import subprocess
import pandas as pd

MAX_FRAMES = 255


def get_next_id_label():
    file_path = 'WLASL_v0.3.json'
    with open(file_path) as f:
        content = json.load(f)

    for ent in content:
        label = ent['gloss']
        for inst in ent['instances']:
            yield inst['video_id'], label


def get_nb_frames(fname):
    nb_frames = -1
    if os.path.isfile(fname):
        result = subprocess.check_output(
            f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{fname}"',
            shell=True).decode()
        if result:
            fields = json.loads(result)['streams'][0]
            nb_frames = int(fields['nb_frames'])

    return nb_frames


def get_video_stats(workdir):
    """ Results below since this takes a while to run
    count    17511.000000
    mean        70.955913
    std         26.044523
    min         13.000000
    25%         51.000000
    50%         71.000000
    75%         87.000000
    max        255.000000
    dtype: float64
    """
    os.chdir(workdir)
    frames = list()
    for id, _ in get_next_id_label():
        path = os.path.join(id + '.mp4')
        nb_frames = get_nb_frames(path)
        if nb_frames > 0:
            frames.append(nb_frames)

    print(pd.Series(frames).describe())


def segment_video(video_path, segment_size=10, frame_size=(640, 640)):
    """Segments a video into discrete pieces of a given size. Will overlap fromes if the segment size is not a
    multiple of the videos total number of frames."""
    nb_frames = get_nb_frames(video_path)
    if segment_size > nb_frames:
        print("WARNING: segment size is larger than video. Not processing video!")
        return []
    cap = cv2.VideoCapture(video_path)
    segments = list()
    frames = list()
    total_processed = 0
    for i in range(nb_frames):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            total_processed += 1
            if len(frames) < segment_size:
                frames.append(frame)
            else:
                segments.append(frames)
                frames = [frame]
    # Sometimes the last few frames of a video are invalid... Thus we need to handle that case. Easiest way is to just
    # ignore the video
    if len(frames) != segment_size and total_processed > segment_size:
        try:
            overlapping_frames = [None] * segment_size
            overlap = segment_size - len(frames)
            overlapping_frames[:overlap] = segments[-1][overlap:]
            overlapping_frames[overlap:] = frames[:]
            segments.append(overlapping_frames)
        except IndexError as e:
            print(f"CAUGHT INDEX ERROR: {e}")
            print("\n PRINTING CONTAINERS")
            print(f"\n\nsegments: {segments}")
            print(f"\n\nframes: {frames}")
            print(f"\n\noverlapping frames: {overlapping_frames}")
            exit(1)
    return segments


def write_video(frame_array, path_out, size=(640, 640), fps=25):
    out = cv2.VideoWriter(path_out, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=size, isColor=True)
    for i in range(len(frame_array)):
        frame = frame_array[i]
        out.write(frame)
    out.release()


def pad_video(video_path, frame_size=(640, 640)):
    cap = cv2.VideoCapture(video_path)
    frames = list()
    has_frames = True
    for i in range(MAX_FRAMES):
        # Capture all valid frames
        if has_frames:
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.resize(frame, frame_size))
            else:
                frames.append(np.zeros((frame_size[0], frame_size[1], 3)))
                has_frames = False
        # Pad with black frames until we reach MAX_FRAMES
        else:
            frames.append(np.zeros((frame_size[0], frame_size[1], 3)))
    return frames


def get_frames(video_path, frame_size=(640, 640)):
    cap = cv2.VideoCapture(video_path)
    frames = list()
    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.resize(frame, frame_size))
    return frames


if __name__ == '__main__':
    segmenting = False
    padding = False
    if segmenting:
        segment_sizes = [10, 15, 25, 71]
        for size in segment_sizes:
            for id, label in get_next_id_label():
                path = os.path.abspath(os.path.join('../data', f'{id}.mp4'))
                if os.path.exists(path):
                    segments = segment_video(path, segment_size=size)
                    if segments:
                        for i, segment in enumerate(segments):
                            segment_name = f'{id}_{i}.mp4'
                            out_dir = os.path.join('../processed_data', str(size), label, id)
                            if not os.path.exists(out_dir):
                                os.makedirs(out_dir)
                            out_path = os.path.abspath(os.path.join(out_dir, segment_name))
                            write_video(segment, out_path)
    elif padding:
        for id, label in get_next_id_label():
            path = os.path.abspath(os.path.join('../data', f'{id}.mp4'))
            if os.path.exists(path):
                frames = pad_video(path)
                out_dir = os.path.join('../processed_data/padded_videos', label)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_path = os.path.abspath(os.path.join(out_dir, f'{id}.mp4'))
                write_video(frames, out_path)
    else:
        for id, label in get_next_id_label():
            path = os.path.abspath(os.path.join('../data', f'{id}.mp4'))
            if os.path.exists(path):
                frames = get_frames(path)
                out_dir = os.path.join('../processed_data/scaled_videos', label)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_path = os.path.abspath(os.path.join(out_dir, f'{id}.mp4'))
                write_video(frames, out_path)
