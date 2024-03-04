# TIME COMPLEXITY: O(n * m)
# n is number of samples and m is n_clusters.

import numpy
import math
import copy
from KMeans_tools import K_Means_tools as tools
from KMeans_tools import NotFittedError as err_fit

class KMeans(tools):
	'''
	K-Means algorithm

	input parameters:
		n_cluster: define the number of cluster you want.
		init: define the initialization methord of centroids.
		n_init: number of centroids initialize and performs convergence on data, best one is saved.
		max_iter: determine the max number of iteration should be performed while convergence of centroids.
		tol: 
		random_state: 
		algorithm: only lloyd algorithm is available.

	attributes: 
		n_cluster: number of clusters.
		cluster_centers: co-ordinates of cluster centers.
 
	example:
		data = numpy.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
		estimator = KMeans(2, 'k-means++', random_state=42)
		estimator.fit(data)
		print(estimator.cluster_centers)
		print(estimator.predict([[0, 0], [3, 6]]))

	More: 
		https://www.github.com/AdarshVishwakarma12/machine-learning-algorithms/machine-learning-algorithms.doc
	'''
	def __init__(self, \
		n_cluster : int = 8, \
		init : ["random", "k-means++"] = "k-means++", \
		n_init : 'auto' or int = 'auto', \
		max_iter : int = 300, \
		tol : float = 0.0001, \
		verbose : int = 0, \
		random_state: int = None, \
		algorithm : ["lloyd", "elkan"] = "lloyd" \
		) -> None:
		
		self.n_cluster = n_cluster
		self.init = init
		self.n_init = n_init
		self.max_iter = max_iter
		self.tol = tol
		self.verbose = verbose
		self.random_state = random_state
		self.algorithm = algorithm
		self.trained = False;


	def fit(self, x : numpy.array) -> None:

		## checking data validity!
		try: 
			assert type(x) == numpy.ndarray
		except: 
			x = numpy.array(x)

		try: 
			assert self.init in ["random", "k-means++"]
		except: 
			raise ValueError(f"init should be either 'k-means++', 'random', got {self.init} instead.")

		try: assert self.algorithm == "lloyd"
		except: 
			if(self.algorithm == 'elkan'): 
				raise ValueError(f"only lloyd algorithm available")
			else: 
				raise ValueError(f'algorithm={self.algorithm}, should be lloyd')
		

		try: assert self.n_init == 'auto' or type(self.n_init) == int
		except: raise Exception(f"n_init : {self.n_init}, should be 'auto' or int value")

		try: assert len(x.shape) == 2;
		except: 
			if(len(x.shape) < 2):
				raise ValueError\
				('''ValueError: Expected 2D array, got 1D array instead:\n\
				Reshape your data using array.reshape(-1, 1) [for Single Feature]\n\
				Reshape your data using array.reshape(1, -1) [for Single Sample]''')
			else:
				raise ValueError(f"Found array with dim {len(x.shape)}. Estimator except <= 2.")			

		try: 
			assert self.n_cluster <= x.shape[0]
		except: 
			raise ValueError(f'n_samples={x.shape[0]}, should be >= n_clusters={self.n_cluster}.')

		try: 
			assert self.random_state == None or type(self.random_state) == int
		except: 
			raise ValueError(f"random_state={self.random_state}, should be {None} or {int} value")

		self.n_features = x.shape[1]
		self.cluster_centers = None
		x = copy.deepcopy(x)

		if(self.n_init == 'auto'):
			if(self.init == 'random'): self.n_init = 10;
			elif(self.init == 'k-means++'): self.n_init = 1

		if(self.random_state!=None):
				numpy.random.seed(self.random_state)

		# train the estimator.
		KMeans._fit(self, x)

		self.trained = True;
		if(self.verbose):
			print("Training Success!")
		return;

	@staticmethod
	def _fit(self, x : numpy.ndarray) -> None:

	    curr_n_init_centroid_loc = None
	    curr_n_init_loss = None # inertia

	    for _i in range(0, self.n_init, 1):

	        ## Initializing centroids
	        centroid_loc = tools.initialize_tool(init = self.init, n_cluster = self.n_cluster, data = x)

	        total_loss = 0

	        ## Performing convergence
	        for i in range(0, self.max_iter, 1):
	            tmp_prev_cluster_loc = centroid_loc

	            ## finding neighrest elements and update the centroids.
	            _, tmp_loss = tools.find_closest(centroid_loc, x, True)
	            total_loss += tmp_loss

	            ## if no change occurs in centroid -> break;
	            if (tmp_prev_cluster_loc == centroid_loc).all(): 
	            	break

	        if(curr_n_init_loss == None or total_loss < curr_n_init_loss):
	            curr_n_init_centroid_loc = centroid_loc
	            curr_n_init_loss = total_loss

	    self.cluster_centers = curr_n_init_centroid_loc
	    return

	def transform(self, X : numpy.ndarray):
		'''
			returns the distance of data samples to all clusters
			test = numpy.array([[0, 1], [3, 1], [-1, -9], [9, 6]])
			print(estimator.transform(test))
		'''

		# Data Validity check!
		try: 
			assert self.trained
		except:
			raise err_fit("This KMeans instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
		
		try: 
			assert len(X.shape) == 2;
		except: 
			if(len(X.shape) < 2):
				raise ValueError\
				('''ValueError: Expected 2D array, got 1D array instead:\n\
				Reshape your data using array.reshape(-1, 1) [for Single Feature]\n\
				Reshape your data using array.reshape(1, -1) [for Single Sample]''')
			else:
				raise ValueError(f"Found array with dim {len(X.shape)}. Estimator except <= 2.")

		try:
			assert X.shape[1] == self.n_features
		except:
			raise ValueError(f"X has {X.shape[1]} features, but KMeans is expecting {self.cluster_centers.shape[1]} features as input.")

		return tools.estimate(X, self.cluster_centers, do = 'transform');


	def predict(self,  X : numpy.ndarray) -> numpy.ndarray:
		'''
			returns the closest cluster to the data sample
			test = numpy.array([[0, 1], [3, 1], [-1, -9], [9, 6]])
			print(estimator.predict(test))
		'''

		# Data Validity check
		try: 
			assert self.trained
		except:
			raise err_fit("This KMeans instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
		
		try: 
			assert len(X.shape) == 2;
		except: 
			if(len(X.shape) < 2):
				raise ValueError\
				('''ValueError: Expected 2D array, got 1D array instead:\n\
				Reshape your data using array.reshape(-1, 1) [for Single Feature]\n\
				Reshape your data using array.reshape(1, -1) [for Single Sample]''')
			else:
				raise ValueError(f"Found array with dim {len(X.shape)}. Estimator except <= 2.")

		try:
			assert X.shape[1] == self.n_features
		except:
			raise ValueError(f"X has {X.shape[1]} features, but KMeans is expecting {self.cluster_centers.shape[1]} features as input.")

		return tools.estimate(X, self.cluster_centers, do = 'predict');


	def fit_transform(self, x : numpy.array):
		x.fit()
		return x.transform()

# TEST CODE
# data = numpy.array([[ 0.,  0.],
#         [ 1.,  0.],
#         [ 1.,  1.],
#         [ 0., 45.],
#         [ 1., 46.],
#         [ 3., 43.],
#         [ 2., 44.],
#         [45.,  1.],
#         [47.,  0.],
#         [46.,  0.],
#         [46.,  1.],
#         [32., 32.],
#         [34., 33.],
#         [31., 33.]], dtype=numpy.float32)

# estimator = KMeans(4, random_state=42, verbose=1)
# estimator.fit(data)
# print(estimator.cluster_centers)
# print(estimator.predict(data))
# print(estimator.transform(data))