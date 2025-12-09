ssh user02@152.118.31.58
pass: user0222

kubectl apply -f pods-gpu02.yaml
kubectl apply -f pods-gpu03.yaml

kubectl get pods

kubectl describe pod user02-gpu-02-cuda-azzam

kubectl exec -it user02-gpu-02-cuda-azzam /bin/bash
kubectl exec -it user02-gpu-03-cuda-azzam /bin/bash

kubectl delete pod user02-gpu-03-cuda-azzam

cd /var/nfs/azzam

//pull results from remote
scp user02@152.118.31.24:/mnt/sharedfolder/user02/matmul-uas/hasil_eksperimen_matmul-gpu02.txt ./ 

//push code to remote
scp ./run_mm_experiments.sh user02@152.118.31.24:/mnt/sharedfolder/user02/matmul-uas/


kubectl get pods
kubectl delete pods [pod]
kubectl apply -f [script]-yam]
kubectl exec -it [pod] /bin/bash
nvidia-smi
nvidia-smi --query-gpu=name --format=csv,noheader
cd /var/nfs
nvcc -o cublas.o cublas.cu -Icublas
nvcc code.cu -o code.o
•/code.o







