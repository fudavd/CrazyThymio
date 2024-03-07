#!/bin/bash

sudo date -s "$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d' ' -f5-8)Z"

sudo apt-get update --fix-missing -y && sudo apt-get upgrade -y

sudo apt install xrdp -y
sudo apt install libatlas-base-dev -y
sudo apt install flatpak -y
sudo flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
sudo flatpak install org.mobsya.ThymioSuite -y

echo -e ‘SUBSYSTEM==“usb”, ATTRS{idVendor}==“0617", ATTRS{idProduct}==“000a”, MODE=“0666"\nSUBSYSTEM==“usb”, ATTRS{idVendor}==“0617", ATTRS{idProduct}==“000c”, MODE=“0666"’ | sudo tee /etc/udev/rules.d/99-mobsya.rules

sudo udevadm control --reload-rules

cd ~/Desktop
mkdir crazy_thymio
cd crazy_thymio
git clone https://github.com/bitcraze/crazyflie-lib-python.git
python -m venv .venv
source .venv/bin/activate
pip3 install tdmclient
cd crazyflie-lib-python
pip install -e .
cd ..
sudo groupadd plugdev
sudo usermod -a -G plugdev $USER

cat <<EOF | sudo tee /etc/udev/rules.d/99-bitcraze.rules > /dev/null
# Crazyradio (normal operation)
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="7777", MODE="0664", GROUP="plugdev"
# Bootloader
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="0101", MODE="0664", GROUP="plugdev"
# Crazyflie (over USB)
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0664", GROUP="plugdev"
EOF


current_ip=$(hostname -I | awk '{print $1}')
gateway=$(ip route | grep default | awk '{print $3}')
dns_servers=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}' | tr '\n' ' ')

# Set the desired static IP address, gateway, and DNS servers
static_ip="$current_ip/24"

sudo sed -i '/interface wlan0/,/^$/ s/^$/static ip_address='"$static_ip"'\nstatic routers='"$gateway"'\nstatic domain_name_servers='"$dns_servers"'\n/' /etc/dhcpcd.conf

echo $current_ip
read -p "Press key to reboot.. " -n1 -s

sudo reboot
