set -x
set +e

error () {
    echo "ERROR: $1"
    exit 1
}


# Activate service account
gcloud auth activate-service-account argo-workload@ddag-gke.iam.gserviceaccount.com --key-file=/ddag-gke-compute-admin/key.json --project=ddag-gke

# Get cluster credentials
gcloud container clusters get-credentials alphafold-dev --zone us-central1-c --project ddag-gke

# Get SCHRODINGER path
stable_build='suite2023-3-build081'
export SCHRODINGER="/mnt/squashsuites/${stable_build}"