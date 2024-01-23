# TorchBench Infra Configuration on Google Cloud Platform

It defines the specification of infrastruture used by TorchBench CI.
The Infra is a Kubernetes cluster built on top of Google Cloud Platform.


## Step 1: Create the cluster and install the ARC Controller

```
# Get credentials for the cluster so that kubectl could use it
gcloud container clusters get-credentials --location us-central1 torchbench-a100-cluster-1

# Install the ARC controller
NAMESPACE="arc-systems"
helm install arc \
    --namespace "${NAMESPACE}" \
    --create-namespace \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller
```

## Step 2: Create secrets and assign it to the namespace

The secrets need to be added to both `arc-systems` and `arc-runners` namespaces.

```
# Set GitHub App secret
kubectl create secret generic arc-secret \
   --namespace=arc-runners \
   --from-literal=github_app_id=<GITHUB_APP_ID> \
   --from-literal=github_app_installation_id=<GITHUB_APP_INSTALL_ID> \
   --from-file=github_app_private_key=<GITHUB_APP_PRIVKEY_FILE>

# Alternatively, set classic PAT
kubectl create secret generic arc-secret \
   --namespace=arc-runners \
   --from-literal=github_token="<GITHUB_PAT>" \
```

To get, delete, or update the secrets:

```
# Get
kubectl get -A secrets
# Delete
kubectl delete secrets -n arc-runners arc-secret
# Update
kubectl edit secrets -n arc-runners arc-secret
```

## Step 3: Install runner scale set

```
INSTALLATION_NAME="a100-runner"
NAMESPACE="arc-runners"
GITHUB_SECRET_NAME="arc-secret"
helm install "${INSTALLATION_NAME}" \
    --namespace "${NAMESPACE}" \
    --create-namespace \
    -f values.yaml \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set
```

To upgrade or uninstall the runner scale set:

```
# command to upgrade
helm upgrade --install a100-runner -n arc-runners -f ./values.yaml oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

# command to uninstall
helm uninstall -n arc-runners a100-runner
```

## Step 4: Install NVIDIA driver on the K8s host machine

```
kubectl apply -f daemonset.yaml
```

When the host machine runs Ubuntu, use the following command to find all available driver versions:

```
gsutil ls gs://ubuntu_nvidia_packages/

# For example:
# gsutil ls gs://ubuntu_nvidia_packages/nvidia-driver-gke_jammy-5.15.0-1048-gke-535.104.05_amd64.deb
```

