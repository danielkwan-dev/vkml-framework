#!/usr/bin/env bash
# Build and test dependencies on Ubuntu 24.04. Used by CI and by tools/ci.Dockerfile,
# so the two can't drift apart.
#
#   compiler + build:   build-essential cmake ninja-build git
#   shader compiler:    glslc
#   Vulkan at runtime:  libvulkan1 (loader), mesa-vulkan-drivers (lavapipe, a
#                       software Vulkan device, so tests run without a GPU),
#                       vulkan-validationlayers
set -euo pipefail

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

$SUDO apt-get update
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git ca-certificates \
    glslc \
    libvulkan1 mesa-vulkan-drivers vulkan-validationlayers
