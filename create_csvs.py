import csv
import os

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
