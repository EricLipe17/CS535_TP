import csv
import json
import os

def get_next_id_label():
    file_path = 'WLASL_v0.3.json'
    with open(file_path) as f:
        content = json.load(f)

    for ent in content:
        label = ent['gloss']
        for inst in ent['instances']:
            yield inst['video_id'], label, inst['split']

if __name__ == '__main__':
    segmenting = False
    padding = False
    if segmenting:
        for i in [10, 15, 25, 71]:
            with open(f'{i}_frame_segments.csv', 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                # This is OS dependent! Fix if you're on unix/linux
                for root, dirs, files in os.walk(f"processed_data\\{i}"):
                    if files:
                        row = list()
                        root.replace('/', '\\')
                        row.append(root)
                        for file in files:
                            row.append(file)
                        label = root.split('\\')[-2]
                        row.append(label)
                        writer.writerow(row)
    elif padding:
        file_map = dict()
        for root, dirs, files in os.walk(f"../processed_data/padded_videos"):
            if files:
                for file in files:
                    f_id = file.split('.')[0]
                    file_map[f_id] = f'{root}/{file}'.replace('\\', '/')

        train_csv = open(f'padded_videos_train.csv', 'w')
        test_csv = open(f'padded_videos_test.csv', 'w')
        train_writer = csv.writer(train_csv, delimiter=',')
        test_writer = csv.writer(test_csv, delimiter=',')
        for id, label, split in get_next_id_label():
            if id in file_map:
                path = file_map[id]
                row = [path, label]
                if split == 'train':
                    train_writer.writerow(row)
                else:
                    test_writer.writerow(row)
    else:
        train_csv = open(f'videos_train.csv', 'w')
        test_csv = open(f'videos_test.csv', 'w')
        train_writer = csv.writer(train_csv, delimiter=',')
        test_writer = csv.writer(test_csv, delimiter=',')

        for id, label, split in get_next_id_label():
            path = f'processed_data/scaled_videos/{label}/{id}.mp4'
            if os.path.exists(f'./data/{id}.mp4'):
                row = [path, label]
                if split == 'train':
                    train_writer.writerow(row)
                else:
                    test_writer.writerow(row)

