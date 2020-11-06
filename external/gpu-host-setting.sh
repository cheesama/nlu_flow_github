#install python 3.8 version
sudo apt-get install build-essential checkinstall
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
cd Python-3.8.5
sudo ./configure --enable-optimizations
sudo make altinstall

#set default python version as 3
sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.8 1

#set default pip version as pip3
sudo apt-get update && sudo apt-get install -y python3-dev python3-pip python-setuptools

#install cuda
# following link(https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130)

#docker install
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install -y docker-ce

#add permission for installing packages
sudo chmod 777 /usr/local/bin
#sudo chmod 777 /var/run/docker.sock
sudo chmod 777 -R /var/run/
sudo chmod 777 -R /usr/local/lib/python3.6/dist-packages/
