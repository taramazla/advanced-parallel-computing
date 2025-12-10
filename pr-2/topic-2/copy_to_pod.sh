#!/bin/bash

# Copy topic-3 folder to GPU server
# The files will be placed in /tmp/topic-3 first, then you can move them to /var/nfs from inside the pod
# Usage: ./copy_to_pod.sh

REMOTE_HOST="user05@152.118.31.58"
REMOTE_PORT="22"
TEMP_DEST="/tmp/topic-2"

echo "Copying files to ${REMOTE_HOST}:${TEMP_DEST}..."
echo "After copying, you can move them to /var/nfs from inside the pod/container"
echo ""

# Create temp directory on remote server
ssh -p ${REMOTE_PORT} ${REMOTE_HOST} "mkdir -p ${TEMP_DEST}"

# Copy files using scp
scp -P ${REMOTE_PORT} *.c *.cu *.sh ${REMOTE_HOST}:${TEMP_DEST}/

echo ""
echo "Done! Files copied to ${TEMP_DEST}"
echo ""
echo "Next steps:"
echo "1. From your SSH session, run: kubectl exec -it user05-gpu-02-cuda-tara -- /bin/bash"
echo "2. Inside the pod, run: cp -r /tmp/topic-3 /var/nfs/"
echo "   OR if /tmp is not shared, copy files directly from the host to the pod's /var/nfs"
