#!/bin/bash

export CLUSTER_NAME=kubeflow-v3-cluster


# Login to IBM Cloud
make iclogin

# Download and update the ibm cloud cluster config file
ibmcloud ks cluster config --cluster $CLUSTER_NAME
