#!/bin/bash


kind create cluster

export PIPELINE_VERSION=1.8.5
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"

kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80


#Added to VF
#curl -LO https://github.com/kubernetes-sigs/kustomize/releases/download/v3.2.0/kustomize_3.2.0_linux_amd64
#chmod +x kustomize_3.2.0_linux_amd64
#sudo mv kustomize_3.2.0_linux_amd64 /usr/local/bin/kustomize



#minikube start --cpus 4 --memory 12288 --disk-size=120g --extra-config=apiserver.service-account-issuer=api --extra-config=apiserver.service-account-signing-key-file=/var/lib/minikube/certs/apiserver.key --extra-config=apiserver.service-account-api-audiences=api

#minikube start --cpus 4 --memory 12288
#
#curl -LO https://github.com/kubernetes-sigs/kustomize/releases/download/v3.2.0/kustomize_3.2.0_linux_amd64
#chmod +x kustomize_3.2.0_linux_amd64
#sudo mv kustomize_3.2.0_linux_amd64 /usr/local/bin/kustomize
#
#git clone git@github.com:kubeflow/manifests.git
#
#kustomize build example/




#sudo snap install kustomize

#git clone git@github.com:kubeflow/manifests.git

#while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done






#echo "Downloading bootstrapper..."
#curl -O https://raw.githubusercontent.com/kubeflow/kubeflow/v0.2-branch/bootstrap/bootstrapper.yaml
#echo "Downloaded!"
#
#echo "Creating bootstrapper K8S env..."
#kubectl create -f bootstrapper.yaml
#echo "Created!"
#
#echo "Verify the setup worked. Show the namespaces..."
#kubectl get ns
#
#echo "Verify the setup worked. Show the services..."
#kubectl -n kubeflow get svc