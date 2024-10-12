import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Function to generate data
def generate_data(a, b, c, variance, N):
    x = np.linspace(-10, 10, N)
    noise = np.random.normal(0, variance, N)
    y = a * x + b + c * noise
    return x, y

# Streamlit UI
st.title('Simple Linear Regression with Adjustable Parameters')

# Parameters
a = st.sidebar.slider('Slope (a)', -10.0, 10.0, 1.0, step=0.1)
a_input = st.sidebar.number_input('Slope (a) Input', value=a, step=0.1)

b = st.sidebar.slider('Intercept (b)', 0, 100, 50, step=1)
b_input = st.sidebar.number_input('Intercept (b) Input', value=b, step=1)

c = st.sidebar.slider('Noise Coefficient (c)', 0.0, 100.0, 1.0, step=0.1)
c_input = st.sidebar.number_input('Noise Coefficient (c) Input', value=c, step=0.1)

variance = st.sidebar.slider('Variance of Noise', 0.0, 10.0, 1.0, step=0.1)
variance_input = st.sidebar.number_input('Variance of Noise Input', value=variance, step=0.1)

N = st.sidebar.slider('Number of Points (N)', 10, 1000, 100, step=10)
N_input = st.sidebar.number_input('Number of Points (N) Input', value=N, step=10)

# Use the most recent values from sliders or inputs
a = a_input
b = b_input
c = c_input
variance = variance_input
N = N_input

# Generate data
x, y = generate_data(a, b, c, variance, N)

# Reshape for sklearn
x_reshaped = x.reshape(-1, 1)
model = LinearRegression()
model.fit(x_reshaped, y)
y_pred = model.predict(x_reshaped)

# Plotting
fig, ax = plt.subplots()
ax.scatter(x, y, label='Data Points')
ax.plot(x, y_pred, color='red', label='Regression Line')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

st.pyplot(fig)

# Display model parameters
st.write(f'Estimated slope (a): {model.coef_[0]:.2f}')
st.write(f'Estimated intercept (b): {model.intercept_:.2f}')
