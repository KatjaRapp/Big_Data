#!/bin/bash
apt-get update
apt-get install python3.6 -y

# add alias for python to bashrc
echo 'alias python=python3' >> ~/.bashrc
source ~/.bashrc

apt-get install unzip

mkdir -p ~/data
python ./google_drive.py "16N-AWvkcvhtQbLT7irpvSHSozJtyvGel" ~/data/stocks.zip
unzip ~/data/stocks.zip