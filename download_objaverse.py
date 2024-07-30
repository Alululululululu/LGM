import objaverse
import multiprocessing
import random
import trimesh
import shutil
import os
import json


processes = multiprocessing.cpu_count()

random.seed(42)

uids = objaverse.load_uids() 
#first_100_uids = uids[:100] 

target_folder = 'objaverse_data'
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

#Download the Objects:
objects = objaverse.load_objects(
    uids=uids,
    download_processes=processes
)

for uid, path in objects.items():
    new_path = os.path.join(target_folder, os.path.basename(path))
    shutil.move(path, new_path)
    objects[uid] = new_path

for uid, path in objects.items():
    print(f"UID: {uid}, New Path: {path}")

mesh = trimesh.load(list(objects.values())[0])

if not os.path.exists('annotations'):
    os.makedirs('annotations')

# annotations
annotations = objaverse.load_annotations()
# annotations = objaverse.load_annotations(uids=uids[:100])


# save
with open('annotations/annotations.json', 'w') as f:
    json.dump(annotations, f, indent=4)

