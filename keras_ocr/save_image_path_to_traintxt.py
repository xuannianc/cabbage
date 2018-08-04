from imutils import paths
import os
import glob
import csv

# 10,11,12 分别对应 年,月,日
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
lines = []
with open('train.csv', 'w') as f:
    for image_path in glob.glob('/home/adam/Pictures/vat_dates/*.jpg'):
        image_file = os.path.split(image_path)[1]
        vat_date_value = []
        vat_date = image_file.split('_')[2]
        # [2,0,1,7,1,2,1,4]
        for c in vat_date:
            vat_date_value.append(c)
        # [2,0,1,7,10,1,2,1,4]
        vat_date_value.insert(4, str(10))
        # [2,0,1,7,10,1,2,11,1,4]
        vat_date_value.insert(7, str(11))
        # [2,0,1,7,10,1,2,11,1,4,12]
        vat_date_value.append(str(12))
        line = list()
        line.append(image_path)
        line.extend(vat_date_value)
        line.append('\n')
        str_line = ','.join(line)
        lines.append(str_line)
    f.writelines(sorted(lines))
