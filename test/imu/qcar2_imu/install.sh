#!/bin/bash
set -e

echo "=============================="
echo " QCar2 IMU Package Installer"
echo "=============================="

echo "[1/4] Configuring Quanser repo..."
cd ~
wget --no-cache https://repo.quanser.com/debian/release/config/configure_repo.sh
chmod u+x configure_repo.sh
./configure_repo.sh
rm -f ./configure_repo.sh

echo "[2/4] Updating apt..."
sudo apt update

echo "[3/4] Installing Quanser Python packages..."
sudo apt install -y \
    python3-quanser-common \
    python3-quanser-communications \
    python3-quanser-devices \
    python3-quanser-hardware \
    python3-quanser-multimedia

echo "[4/4] Installing numpy..."
pip3 install numpy

echo ""
echo "Installation complete."
echo "Run: python3 ~/qcar2_imu/scripts/01_verify_connection.py"
