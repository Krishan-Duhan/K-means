import sys
import numpy as np
import math

if len(sys.argv) != 5 :
	print('usage : ', sys.argv[0], 'data_file k(num of clusters) sigma output_file')
	sys.exit()


def find_laplacian(X,rows,sig):
	data_size = len(rows)
	distance = np.zeros((data_size,data_size))
	for i in range(data_size):
		for j in range(data_size):
			distance[i,j] = np.linalg.norm(X[i]- X[j])
	
	weight = np.zeros((data_size,data_size))
	degree = np.zeros(data_size)
	for i in range(data_size):
		for j in range(data_size):
			gamma=0.5*math.pow((distance[i,j]/sig),2)
			weight[i,j] = math.exp(-gamma)
			degree[i] = degree[i]+weight[i,j]
	
	norm_laplacian = np.zeros((data_size,data_size))
	for i in range(data_size):
		for j in range(data_size):
			if i==j and degree[i]!=0:
				norm_laplacian[i,j]=1
			elif i!=j and degree[i]!=0 and degree[j]!=0:  
				norm_laplacian[i,j]=(-weight[i,j])/math.sqrt(degree[i]*degree[j])
			else:
				norm_laplacian[i,j]=0
	return weight, degree, norm_laplacian


def spectral_clustering(rows, weight, degree, norm_laplacian):
	data_size = len(rows)
	eigen_vals, eigen_vecs = np.linalg.eig(norm_laplacian)
	#need to select top 2 eigen values and eigen vectors
	idx = np.argsort(eigen_vals)[::-1] # sort in reverse order
	eigen_vals = eigen_vals[idx]
	eigen_vecs = eigen_vecs[:,idx]
	eigen_val_second_small = eigen_vals[-2]
	eigen_vec_second_small = eigen_vecs[:,-2]
	idx = np.argsort(eigen_vec_second_small)[::]
	h=float('inf')
	partition1=None
	partition2=None
	for k in range(len(idx)-1):
		A=set(idx[:k+1])
		B=set(idx[k+1:])
		Va=0
		Vb=0
		Cab=0
		for i in range(data_size):
			if i in A:
				Va=Va+degree[i]
			elif i in B:
				Vb=Vb+degree[i]
			for j in range(data_size):
				if i in A and j in B:
					Cab = Cab+weight[i,j]
		temp = Cab/(float(min(Va,Vb)))
		if temp<h:
			partition1 = list(A)
			partition2 = list(B)
			h = temp
	
	partition1 = np.asarray(rows)[partition1].tolist()
	partition2 = np.asarray(rows)[partition2].tolist()
	return partition1,partition2


def quant_err(X,partition_list):
	error_value = 0.0
	for partition in partition_list:
		data = X[partition]
		centroid = data.mean(axis=0)
		for row in data:
			error_value = error_value+math.pow(np.linalg.norm(row - centroid), 2)
	return error_value


X = np.genfromtxt(sys.argv[1], delimiter = ',', autostrip=True)
sigma = float(sys.argv[3])
partition_list = [list(i for i in range(X.shape[0]))]
partition = partition_list.pop(0)
wt, degree, norm_laplac = find_laplacian(X, partition, sigma)
new_parts = spectral_clustering(partition, wt, degree, norm_laplac)
partition_list.extend(new_parts)

while len(partition_list) < int(sys.argv[2]):
	min_lambda=float('inf')
	min_partition = None
	min_weight = None
	min_degree = None
	min_norm_laplacian = None
	for K in range(len(partition_list)):
		wt, degree, norm_laplac = find_laplacian(X, partition_list[K], sigma)
		eigen_vals, eigen_vecs = np.linalg.eig(norm_laplac)
		idx = np.argsort(eigen_vals)[::-1] # sort in reverse order
		eigen_vals = eigen_vals[idx]
		eigen_val_second_small = eigen_vals[-2]
		if eigen_val_second_small<min_lambda:
			min_degree = degree
			min_norm_laplacian = norm_laplac
			min_partition=K
			min_weight = wt
	partition = partition_list.pop(min_partition)
	new_partiton = spectral_clustering(partition, min_weight, min_degree, min_norm_laplacian)
	partition_list.extend(new_partiton)

cluster_map = np.zeros((X.shape)[0])
for i in range(len(partition_list)):
	for j in partition_list[i]:
		cluster_map[j] = i
		

#np.savetxt(sys.argv[4], np.array(cluster_map).reshape(75,1), delimiter = ',', fmt='%d')
	
print("Quantization err: ", quant_err(X,partition_list))