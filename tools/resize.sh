#!/bin/bash
find . -name "*.jpg" -print0 | xargs -0 mogrify -resize 256x256
