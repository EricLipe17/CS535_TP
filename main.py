from dataset import WLASLSegmentDataset

data_set = WLASLSegmentDataset('', '10_frame_segments.csv')

for label, segments in data_set:
    pass

