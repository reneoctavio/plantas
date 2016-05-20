#!/bin/bash

find . -name "*.jpg" -print0 | xargs -0 mogrify -crop 1536x1536+256+0
