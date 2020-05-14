#!/bin/bash
set -ex

user_id=${USER_ID:-0}
user_name=${USER_NAME:-"root"}

if [ "$user_name" = "root" ] || [ $user_id -eq 0 ]; then
    exec "$@"
    exit 0
fi

if [ $(cat /etc/passwd | cut -f1 -d':' | grep -w "$user_name" -c) -eq 0 ]; then
    useradd -u $user_id -o -m $user_name
    sed -i 's/^##includedir \/etc\/sudoers.d/#includedir \/etc\/sudoers.d/g' /etc/sudoers
    echo "$user_name ALL=(ALL)       NOPASSWD: ALL" > /etc/sudoers.d/$user_name
fi

exec /usr/local/bin/gosu "$user_name" "$@"
