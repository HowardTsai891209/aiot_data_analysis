import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate random points
def generate_points(num_points=600):
    mean = [0, 0]
    covariance = [[10, 0], [0, 10]]  # Variance of 10 for both x1 and x2
    points = np.random.multivariate_normal(mean, covariance, num_points)
    
    # Calculate distances and assign labels
    distances = np.sqrt(np.sum(points**2, axis=1))
    labels = np.where(distances < 4, 0, 1)
    
    return points, labels

# Step 2: Create a Gaussian function for x3
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

# Streamlit interface
def main():
    st.title('Random Point Generation and Visualization')

    # Generate points
    num_points = 600
    points, labels = generate_points(num_points)

    # 2D Scatter Plot
    st.subheader('2D Scatter Plot')
    plt.figure(figsize=(10, 6))
    plt.scatter(points[labels == 0, 0], points[labels == 0, 1], color='blue', label='Y=0 (distance < 4)', alpha=0.6)
    plt.scatter(points[labels == 1, 0], points[labels == 1, 1], color='red', label='Y=1 (distance >= 4)', alpha=0.6)
    plt.title('2D Scatter Plot of Random Points')
    plt.xlabel('X1-axis')
    plt.ylabel('X2-axis')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    st.pyplot(plt)

    # 3D Scatter Plot
    st.subheader('3D Scatter Plot')
    x3 = gaussian_function(points[:, 0], points[:, 1])
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(points[:, 0], points[:, 1], x3, c=labels, cmap='coolwarm', label='Y-labels')
    
    ax.set_title('3D Scatter Plot of Random Points with Gaussian Function')
    ax.set_xlabel('X1-axis')
    ax.set_ylabel('X2-axis')
    ax.set_zlabel('X3-axis (Gaussian function)')
    
    # Add a color bar
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend1)
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()
