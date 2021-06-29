#!/bin/bash -xe

SCRIPT_DIR=$(cd $(dirname $0);pwd)

mpiFCCpx -Nclang -std=c++14 -Kfast -fopenmp -I "${SCRIPT_DIR}/sim_wsn_all/cpp" -I "${SCRIPT_DIR}/sim_wsn_all/cpp/wsn" -I "${SCRIPT_DIR}/caravan-lib" -I "${SCRIPT_DIR}/caravan-lib/json/single_include" -I "${SCRIPT_DIR}/caravan-lib/icecream" search.cpp -o search.out

