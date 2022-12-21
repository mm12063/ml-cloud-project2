#!/bin/bash

export CLUSTER_NAME=kubeflow-vpc
#export CLUSTER_NAME=stock-predictor-clust

# Login to IBM Cloud
make iclogin

# Download and update the ibm cloud cluster config file
ibmcloud ks cluster config --cluster $CLUSTER_NAME