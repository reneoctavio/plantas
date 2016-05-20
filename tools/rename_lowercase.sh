#!/bin/bash

find . -name '*.JPG' -exec bash -c 'mv "$1" "${1/%.JPG/.jpg}"' -- {} \;
