import hashlib
import pandas as pd
import os.path

csv_intenet = '/home/reneoctavio/Documents/Plantas/Plantas+/Internet/plantas-internet.csv'

internet_path = '/home/reneoctavio/Documents/Plantas/Plantas+/Internet'

df = pd.read_csv(csv_intenet)

hashed = []
for idx, row in df.iterrows():
    filepath = os.path.join(internet_path, row['Label'], row['File'])

    hasher = hashlib.md5()
    with open(filepath, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)

    hashed.append(hasher.hexdigest())

    print('File: ' + row['File'] + '   Hash: ' + str(hashed[-1]))

df['MD5-Hash'] = hashed

df.to_csv(os.path.join(internet_path, 'plantas-internet-md5-hashed.csv'), columns=['File', 'Label', 'URL', 'MD5-Hash'])
