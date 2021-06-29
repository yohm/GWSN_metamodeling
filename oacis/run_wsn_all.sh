#!/bin/bash

set -eux
SCRIPT_DIR=$(cd $(dirname $0); pwd)
${SCRIPT_DIR}/../cmake-build-release/wsn.out _input.json
