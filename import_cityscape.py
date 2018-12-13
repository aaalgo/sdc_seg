#!/usr/bin/env python3 
import sys
import os 
import subprocess as sp
from glob import glob 
import cv2
import simplejson as json
import picpac 
from tqdm import tqdm
import numpy as np

json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

LABEL_MAP = {'road': 3,
             'person': 2,
             'rider': 2,
             'persongroup': 2,  # <-- added
             'ridergroup': 2,   # <-- added
             'car': 1,
             'cargroup': 1,     # <-- added
             'truck': 1,
             'truckgroup': 1,
             'bus': 1,
             'caravan': 1,
             'trailer': 1,
             'train': 1,
             'motorcycle': 2,
             'motorcyclegroup': 2, 
             'bicycle': 2,
             'bicyclegroup': 2}
# sort shapes by order, higher order will overwrite lower order in rendering
ORDER_MAP = {3: 0, 1: 1, 2: 2}

# Following are not handled.
# ['sky', 'terrain', 'guard rail', 'bridge', 'static', 'fence', 'vegetation', 'building', 'parking', 'tunnel', 'wall', 'rectification border', 'out of roi', 'traffic light', 'pole', 'traffic sign', 'ground', 'polegroup', 'dynamic', 'sidewalk', 'license plate', 'ego vehicle', 'rail track']

UNKNOWN = {}

 
def import_db (split): #db_path, img_path, labelid_path):
    db_path = 'scratch/cityscape/%s.db' % split
    tasks = glob('data/cityscape/leftImg8bit/%s/*/*.png' % split)
    print('importing %s: %d images' % (split, len(tasks)))
    db = picpac.Writer(db_path,picpac.OVERWRITE)
    for image_path in tqdm(tasks):
        # image_path: data/cityscape/leftImg8bit/train/jena/jena_000050_000019_leftImg8bit.png
        image = cv2.imread(image_path, -1)
        if image is None:
            continue
        if not image.shape[0] > 0:
            continue
        with open(image_path, 'rb') as f:
            image_buf = f.read()
            pass
        if split == 'test': # we don't need to load labels for test images
            db.append(0, image_buf)
            continue
        label_path = image_path.replace('leftImg8bit', 'gtFine').replace('.png', '_polygons.json')
        assert os.path.exists(label_path)
        with open(label_path, 'r') as f:
            anno = json.loads(f.read())
            pass
        H = anno['imgHeight']
        W = anno['imgWidth']

        shapes = []
        for obj in anno['objects']:
            label_txt = obj['label']
            label = LABEL_MAP.get(label_txt, None)
            if label is None:
                UNKNOWN[label_txt] = 1
            else:
                points = []
                for x, y in obj['polygon']:
                    points.append({'x': x / W, 'y': y / H})
                    pass
                order = ORDER_MAP[label]
                shapes.append({'type':'polygon', 'label': label, 'order': order, 'geometry':{'points': points}})
            pass
        shapes.sort(key=lambda x: x['order'])
        label_buffer = json.dumps({'shapes': shapes}).encode('ascii')
        db.append(0, image_buf, label_buffer)
        pass


sp.check_call('mkdir -p ./scratch/cityscape', shell=True)
import_db('train')
import_db('val')
import_db('test')
print("unknown:", UNKNOWN.keys())

