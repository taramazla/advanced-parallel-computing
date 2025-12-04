# advanced-parallel-computing

##  Cluster 8 Node Fasilkom UI
ssh user05@152.118.31.24 -p 22

## Multi-Core Fasilkom UI
ssh user05@152.118.31.61 -p 22


mkdir UTS-Tara
cd UTS-Tara

nano matrix_vector_256.c
nano run-mat-vec.sh

nano matrix_matrix_256.c
nano run-mat-mat.sh

nano results.sh


Matrix-Vector multiplication (row-wise distribution)
Matrix-Matrix multiplication (standard row distribution)
Matrix-Matrix multiplication dengan menggunakan Cannon's algorithm (2D block distribution)


user05@store-01:~/coba3$ mpicc deadlock.c -o deadlock.o
user05@store-01:~/coba3$ mpirun -np 4 ./deadlock.o

ssh user05@152.118.31.58 -p 22

kubectl apply -f pod-name.yaml
kubectl get pods
kubectl delete pods <pod-name>

topic-3 % ./copy_to_pod.sh

kubectl cp /tmp/topic-3 user05-gpu-02-cuda-tara:/var/nfs/topic-3

kubectl exec -it user05-gpu-02-cuda-tara -- /bin/bash


cd pr-2/topic-3

# 1. Build everything
make all

# 2. Run experiments (takes time!)
chmod +x run_experiments.sh
./run_experiments.sh

# 3. Analyze and visualize
python3 analyze_results.py

# 4. Check results
ls experiment_results/
# → PNG plots, LaTeX tables, analysis report