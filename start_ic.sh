#!/bin/bash

# Login to IBM Cloud
make iclogin

# Download and update the ibm cloud cluster config file
ibmcloud ks cluster config --cluster ENTER_CLUSTER_NAME
