from sage.all import *

def extract_submatrix(indices0, indices1, Afull):
	A = matrix(SR, len(indices0), len(indices1))
	
	for i in range(len(indices0)):
		for j in range(len(indices1)):
			A[i, j] = Afull[indices0[i], indices1[j]]
	
	return A

def partial_covariance_matrix(targets, condset, Sigma):
	SigmaX = extract_submatrix(targets, targets, Sigma)
	SigmaXZ = extract_submatrix(targets, condset, Sigma)
	SigmaZ = extract_submatrix(condset, condset, Sigma)
	
	SigmaXgZ = SigmaX  - SigmaXZ*(SigmaZ.inverse())*SigmaXZ.T
	
	return SigmaXgZ

def partial_regression_coefficient(response, predictor, condset, Sigma):
	targets = [response, predictor]

	SigmaX = extract_submatrix(targets, targets, Sigma)
	SigmaXZ = extract_submatrix(targets, condset, Sigma)
	SigmaZ = extract_submatrix(condset, condset, Sigma)
	
	SigmaXgZ = SigmaX  - SigmaXZ*(SigmaZ.inverse())*SigmaXZ.T
	
	return SigmaXgZ[0, 1]/SigmaXgZ[1, 1]