import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

# Title
st.title("Neural Network Demo")

# User Inputs
x1 = st.number_input("Feature 1", value=6.0)
x2 = st.number_input("Feature 2", value=6.0)
x3 = st.number_input("Feature 3", value=8.0)
x4 = st.number_input("Feature 4", value=2.5)

# Convert input to tensor
X = torch.tensor([[x1, x2, x3, x4]])
y = torch.tensor([[0.0]])

# Model
model = nn.Sequential(
    nn.Linear(4,2),
    nn.Sigmoid(),
    nn.Linear(2,1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train button
if st.button("Train Model"):
    for epoch in range(100):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    st.success("Training Complete!")
    st.write("Final Prediction:", model(X).item())