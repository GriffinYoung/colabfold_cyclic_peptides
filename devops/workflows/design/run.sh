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
export STABLE_BUILD='suite2023-3-build081'
export SCHRODINGER="/mnt/squashsuites/${STABLE_BUILD}"
ls $SCHRODINGER

# Copy the environment from a bucket (or create it if the script hasn't been run before)
source devops/workflows/install_colabdesign.sh

# Run colabfold then bond the termini and protein prep
source devops/workflows/run_cyclic_colabdesign.sh