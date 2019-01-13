#!/bin/bash
apt-get update
apt-get install python3.6 -y

# add alias for python to bashrc
echo 'alias python=python3' >> /home/ubuntu/.bashrc
source /home/ubuntu/.bashrc

apt-get install unzip

mkdir -p /home/ubuntu/data
python ./google_drive.py "16N-AWvkcvhtQbLT7irpvSHSozJtyvGel" /home/ubuntu/data/stocks.zip
unzip /home/ubuntu/data/stocks.zip