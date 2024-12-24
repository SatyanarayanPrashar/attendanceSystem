import numpy as np

def perform_pca(data_matrix, num_components=50):
    """
    Performs PCA on the face data matrix.

    Parameters:
    - data_matrix (numpy.ndarray): Matrix of flattened face images (num_samples x num_features).
    - num_components (int): Number of principal components (eigenfaces) to retain.

    Returns:
    - eigenfaces (numpy.ndarray): Principal components (num_components x num_features).
    - mean_face (numpy.ndarray): Mean face vector (1 x num_features).
    - projections (numpy.ndarray): Data projections onto principal components (num_samples x num_components).
    """
    # Calculate the mean face
    mean_face = np.mean(data_matrix, axis=0)
    
    # Center the data by subtracting the mean face
    centered_data = data_matrix - mean_face
    
    # Compute covariance matrix
    covariance_matrix = np.dot(centered_data, centered_data.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Compute eigenfaces
    eigenfaces = np.dot(centered_data.T, eigenvectors)
    eigenfaces = eigenfaces[:, :num_components]
    
    # Normalize eigenfaces
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)
    
    # Project the data onto the eigenfaces
    projections = np.dot(centered_data, eigenfaces)
    
    return eigenfaces, mean_face, projections

if __name__ == "__main__":
    from load_data import load_face_data
    
    data_matrix, labels, names = load_face_data()
    eigenfaces, mean_face, projections = perform_pca(data_matrix)
    print("PCA completed. Eigenfaces and projections calculated.")
