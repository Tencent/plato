#!/bin/bash

set -ex

. /etc/os-release

debian_packages=(
    unzip
    zip
    gcc
    g++
    make
    autoconf
    automake
    libtool
    pkg-config
    cmake
    wget
    curl
    libnuma-dev
    zlib1g-dev
)

redhat_packages=(
    which
    numactl-devel
    unzip
    zip
    zlib-devel
    gcc
    gcc-c++
    libstdc++-static
    make
    libtool
    wget
    curl
    cmake
)

function install_bazel() {
    BAZEL_INSTALLED=True
    command -v bazel > /dev/null 2>&1 || { BAZEL_INSTALLED=False; }
    if [ $BAZEL_INSTALLED = True ]; then
        echo "bazel has already installed."
        return;
    fi
    BAZEL_VERSION=1.2.0
    BAZEL_INSTALLER_FIL=bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
    BAZEL_INSTALLER=/tmp/$BAZEL_INSTALLER_FIL
    curl -o $BAZEL_INSTALLER -sSL "https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/$BAZEL_INSTALLER_FIL"
    chmod +x $BAZEL_INSTALLER
    $BAZEL_INSTALLER
    rm $BAZEL_INSTALLER
}

if [ "$ID" = "ubuntu" ] || [ "$ID" = "debian" ]; then
    apt-get install -y "${debian_packages[@]}"
elif [ "$ID" = "centos" ] || [ "$ID" = "fedora" ] || [ "$ID" = "tlinux" ]; then
    if [ "$ID" = "centos" ] && [ "$VERSION_ID" = "8" ]; then # centos 8
        dnf --enablerepo=PowerTools install -y "${redhat_packages[@]}"
    else
        yum install -y "${redhat_packages[@]}"
    fi
else
    echo "Your system ($ID) is not supported by this script. Please install dependencies manually."
    exit 1
fi

install_bazel
