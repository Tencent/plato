
#!/bin/bash

set -ex

GOSU_VERSION=1.11

curl -o /usr/local/bin/gosu -sSL "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-amd64"
chmod +x /usr/local/bin/gosu
gosu --version
gosu nobody true
