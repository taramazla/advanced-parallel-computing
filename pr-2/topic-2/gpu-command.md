ssh user02@152.118.31.58

kubectl apply -f pods-gpu02.yaml

kubectl get pods

kubectl describe pod user02-gpu-02-cuda-azzam

kubectl exec -it user02-gpu-02-cuda-azzam /bin/bash
kubectl exec -it user02-gpu-03-cuda-azzam /bin/bash

kubectl delete pod user02-gpu-03-cuda-azzam

cd /var/nfs/azzam

scp user02@152.118.31.24:/mnt/sharedfolder/user02/azzam/hasil_eksperimen_matmul.txt ./ 

