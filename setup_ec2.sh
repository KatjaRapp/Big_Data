#!/bin/bash
apt-get update
apt-get install python3.6 -y

# add alias for python to bashrc
echo 'alias python=python3' >> ~/.bashrc
source ~/.bashrc

apt-get install unzip