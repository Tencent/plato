FROM debian:9
MAINTAINER platoteam

COPY install-dependencies.sh /usr/local/bin/
RUN set -ex && \
    apt-get update && \
    apt-get install -y sudo && \
    install-dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

COPY install-gosu.sh /usr/local/bin/
RUN install-gosu.sh

RUN mkdir /data
WORKDIR /data

COPY entrypoint.sh /usr/local/bin/
ENTRYPOINT ["entrypoint.sh"]
CMD ["/bin/bash"]
