# The CI environment as a container, for reproducing CI locally:
#
#   docker build -t vkml-ci -f tools/ci.Dockerfile tools
#   docker run --rm -v "$PWD:/src:ro" vkml-ci
#
# The source is mounted read-only and copied inside, so the build never
# writes into (or reads a stale build/ from) the host checkout.
FROM ubuntu:24.04

COPY install-deps-ubuntu.sh /tmp/
RUN bash /tmp/install-deps-ubuntu.sh && rm -rf /var/lib/apt/lists/*

CMD ["bash", "-c", "cp -r /src /work && rm -rf /work/build && cd /work && cmake --preset ci && cmake --build --preset ci && ctest --preset ci && ./build/ci/apps/vkml-info && cmake -S tools/consumer -B build/consumer -G Ninja && cmake --build build/consumer && ./build/consumer/quickstart"]
