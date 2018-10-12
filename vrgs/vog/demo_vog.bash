#!/bin/bash

echo ''
echo -e "\e[34m======== Steps 1 & 2: Subgraph Generation and Labeling  ==========\e[0m"
matlab -r -nodesktop run_structureDiscovery
echo ''
echo 'Structure discovery finished.'

unweighted_graph='DATA/karate.g'
# model='DATA/karate.model'
# modelFile='karate.model'