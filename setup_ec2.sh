#!/bin/bash
apt-get update
apt-get install python3.6 -y
apt-get install python3-pip -y

pip3 install numpy
pip3 install matplotlib
pip3 install pandas
pip3 install sklearn

# add alias for python3 and pip3 to bashrc
echo 'alias python=python3' >> /home/ubuntu/.bashrc
echo 'alias pip=pip3' >> /home/ubuntu/.bashrc
source /home/ubuntu/.bashrc

apt-get install unzip -y

mkdir -p /home/ubuntu/data
python3 ./google_drive.py "16N-AWvkcvhtQbLT7irpvSHSozJtyvGel" /home/ubuntu/data/stocks.zip
unzip /home/ubuntu/data/stocks.zip -d /home/ubuntu/data
chmod -R 777 /home/ubuntu/data

# set up git
git config --global user.name "Katja Rapp"
git config --global user.email rapp.katja@googlemail.com

# execute python script
python3 ~/Big_Data/predictions.py