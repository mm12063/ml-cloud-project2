#!/bin/bash

kustomize build ibm-manifests-160/common/cert-manager/cert-manager/base | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/cert-manager/kubeflow-issuer/base | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/istio-1-14/istio-crds/base | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/istio-1-14/istio-namespace/base | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/istio-1-14/istio-install/base | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/dex/overlays/istio | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/oidc-authservice/base | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/knative/knative-serving/overlays/gateways | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/istio-1-14/cluster-local-gateway/base | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/knative/knative-eventing/base | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/kubeflow-namespace/base | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/kubeflow-roles/base | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/istio-1-14/kubeflow-istio-resources/base | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/apps/pipeline/upstream/env/cert-manager/platform-agnostic-multi-user | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/apps/pipeline/upstream/env/platform-agnostic-multi-user-pns | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/contrib/kserve/kserve | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/contrib/kserve/models-web-app/overlays/kubeflow | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/apps/katib/upstream/installs/katib-with-kubeflow | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/apps/centraldashboard/upstream/overlays/kserve | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/apps/admission-webhook/upstream/overlays/cert-manager | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/apps/jupyter/notebook-controller/upstream/overlays/kubeflow | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/apps/jupyter/jupyter-web-app/upstream/overlays/istio | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/apps/profiles/upstream/overlays/kubeflow | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/apps/volumes-web-app/upstream/overlays/istio | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/apps/tensorboard/tensorboards-web-app/upstream/overlays/istio | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/apps/tensorboard/tensorboard-controller/upstream/overlays/kubeflow | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/apps/training-operator/upstream/overlays/kubeflow | kubectl apply -f -
sleep 10;
kustomize build ibm-manifests-160/common/user-namespace/base | kubectl apply -f -
sleep 10;