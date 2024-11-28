#!/bin/bash

curl -L -o models.tgz -H \
  "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) " \
  https://datadryad.org/stash/downloads/file_stream/242856

tar -zxvf models.tgz

#rm models.tgz
