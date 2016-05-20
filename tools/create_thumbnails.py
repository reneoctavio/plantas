from PIL import Image
import os
import os.path

import random

core_path = '/Users/reneoctavio/Documents/Plantas+/Core'
internet_path = '/Users/reneoctavio/Documents/Plantas+/Internet'

files = []
for label in os.listdir(core_path):
    if not '.' in label:
        path = os.path.join(core_path, label)
        dir_files = [f for f in os.listdir(path) if not os.path.isdir(f)]
        chosen_file = random.choice(dir_files)
        files.append(os.path.join(path, chosen_file))

# Get images size
img = Image.open(files[0], 'r')
img_w, img_h = img.size
img.close()

# Create a 10x5 background
b_width = img_w / 10
b_height = img_h / 10
background = Image.new('RGBA', (b_width * 10, b_height * 5), (255, 255, 255, 255))


# for hei in range(0, 5):
#     for wid in range(0, 10):
#         f = files[(hei * 10) + wid]
#         print(f)
#
#         off_width = b_width *  wid
#         off_height = b_height * hei
#         offset = (off_width, off_height)
#
#         img = Image.open(f, 'r')
#         img = img.resize((b_width,b_height), Image.ANTIALIAS)
#         background.paste(img, offset)
#
# background.save(os.path.join(core_path, 'core_thumbnail.png'))


files = []
for label in os.listdir(internet_path):
    if not '.' in label:
        path = os.path.join(internet_path, label)
        dir_files = [f for f in os.listdir(path) if not os.path.isdir(f)]
        chosen_file = random.choice(dir_files)
        files.append(os.path.join(path, chosen_file))

for hei in range(0, 5):
    for wid in range(0, 10):
        f = files[(hei * 10) + wid]
        print(f)

        off_width = b_width *  wid
        off_height = b_height * hei
        offset = (off_width, off_height)

        img = Image.open(f, 'r')
        img = img.resize((b_width,b_height), Image.ANTIALIAS)
        background.paste(img, offset)

background.save(os.path.join(internet_path, 'internet_thumbnail.png'))
