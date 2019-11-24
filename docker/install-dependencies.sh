#!/bin/bash

set -ex

. /etc/os-release

debian_packages=(
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
    zip
    unzip
    libnuma-dev
    zlib1g-dev
    python
)

redhat_packages=(
    numactl-devel
    zlib-devel
    gcc
    gcc-c++
    glibc-static
    libstdc++-static
    make
    libtool
    wget
    curl
    cmake
    bazel
)

if [ "$ID" = "ubuntu" ] || [ "$ID" = "debian" ]; then
    apt-get install -y "${debian_packages[@]}"
    # install bazel
    BAZEL_VERSION=1.2.0
    BAZEL_INSTALLER_FIL=bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
    BAZEL_INSTALLER=/tmp/$BAZEL_INSTALLER_FIL
    curl -o $BAZEL_INSTALLER -sSL "https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/$BAZEL_INSTALLER_FIL"
    chmod +x $BAZEL_INSTALLER
    $BAZEL_INSTALLER
    rm $BAZEL_INSTALLER
elif [ "$ID" = "centos" ] || [ "$ID" = "fedora" ]; then
    if [ "$ID" = "fedora" ]; then
        dnf install -y yum-plugin-copr
        dnf copr enable -y vbatts/bazel
        dnf install -y "${redhat_packages[@]}"
    else # centos
        if [ $VERSION_ID = "8" ]; then
            dnf install -y epel-release
            dnf install -y yum-plugin-copr
            dnf copr enable -y vbatts/bazel
            dnf --enablerepo=PowerTools install -y "${redhat_packages[@]}"
        else
            yum install -y epel-release
            yum install -y yum-plugin-copr
            yum copr enable -y vbatts/bazel
            yum install -y "${redhat_packages[@]}"
        fi
    fi
else
    echo "Your system ($ID) is not supported by this script. Please install dependencies manually."
    exit 1
fi
