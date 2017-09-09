# Run install from home directory so that tensofrflow virtualenv will
# be ~/tensorflow.
cd ~

# Bring ubuntu install up to date
sudo apt-get update
sudo apt-get upgrade

# Install ffmpeg
sudo apt-get install ffmpeg

# Install tensorflow for python 3 without GPU suport
sudo apt-get install python3-pip python3-dev python-virtualenv
virtualenv --system-site-packages -p python3 tensorflow
source ~/tensorflow/bin/activate
easy_install -U pip
pip3 install --upgrade tensorflow
source ~/tensorflow/bin/activate

# Install numpy and scipy
sudo pip3 install numpy scipy
