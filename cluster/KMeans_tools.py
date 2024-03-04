import numpy
import math
import copy

class K_Means_tools():
	# Initializing the certroids
	def initialize_tool(**kwargs) -> numpy.ndarray:

		if(kwargs['init'] == "k-means++"):
			data = kwargs['data']
			n_cluster = kwargs['n_cluster']
			clusters = list()
			distance_matrix = [None] * data.shape[0]
			current_cluster = None

			for i in range(n_cluster):
				if(i==0):
					current_cluster = data[numpy.random.randint(0, data.shape[0], size = 1)[0]]
				else:
					current_cluster = data[numpy.argmax(distance_matrix)]

				clusters.append(list(current_cluster))
				# Updating distance matrix
				if(None in distance_matrix):
					distance_matrix = numpy.sum(abs(data - current_cluster), axis=1)
				else:
					distance_matrix = numpy.minimum(distance_matrix, numpy.sum(abs(data - current_cluster), axis=1))
			return numpy.array(clusters)

		elif(kwargs['init'] == "random"):
			data = kwargs['data']
			n_cluster = kwargs['n_cluster']

			clusters_tmp = list()
			ran_tmp = numpy.random.randint(low=0, high=data.shape[0], size=n_cluster)
			
			for i in ran_tmp: clusters_tmp.append(list(data[i]))
				
			return numpy.array(clusters_tmp)

	# lloyd algorithm
	def find_closest(cluster_loc : numpy.ndarray, data : numpy.ndarray, inplace = True) -> None:

		if inplace == False:
		    cluster_loc = copy.deepcopy(cluster_loc)

		dict_ = {}
		num_sample = [1] * len(cluster_loc)
		loss = 0

		# # initialize the dict
		for i in range(0, len(cluster_loc), 1): 
		    dict_.update({i : numpy.broadcast_to([0], shape=data.shape[1])})

		# # data format -> (n_samples, n_features)
		# # centroid format -> (n_cluster, n_features)

		for i in range(0, data.shape[0], 1): # single sample

		    tmp = numpy.sum(abs(data[i] - cluster_loc), axis=1);	
		    idx = numpy.argmin(tmp)
		    loss += tmp[idx]
		    dict_[idx] = dict_[idx] + data[i]
		    num_sample[idx] += 1

		# # scale dict_
		for i in dict_.keys(): 
		    cluster_loc[i] = (cluster_loc[i] + dict_[i]) / num_sample[i]

		return cluster_loc, loss

	def estimate(y : numpy.ndarray, cluster_loc : numpy.ndarray, do : ['predict', 'transform'] = "predict") -> numpy.ndarray:

		res = list()
		for i in y:
			tmp = numpy.sum(abs(i - cluster_loc), axis=1)
			if(do == 'transform'): 
				res.append(tmp)
			elif(do == 'predict'): 
				res.append(tmp.argmin())
			else:
				raise Exception("Unexpected code error")

		return numpy.array(res)

class NotFittedError(Exception):
    pass