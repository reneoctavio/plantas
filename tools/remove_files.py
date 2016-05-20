# Imports
import os
import os.path
import shutil

_PATH  = '/media/reneoctavio/Rene-1/'

# Copy jpeg images to be treated manually
_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(_PATH) for f in fn if f.endswith((":Mac_Metadata", ":AFP_AfpInfo"))]

for _file in _files:
    # Move
    # print(_file)
    os.remove(_file)
