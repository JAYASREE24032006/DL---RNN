# EX-5 : DL- DEVELOPING A RECURRENT NEURAL NETWORK MODEL FOR STOCK PREDICTION

#### NAME : R.JAYASREE
#### R.NO : 212223040074

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
To design and implement a Recurrent Neural Network (RNN) that learns temporal patterns from historical stock closing prices and predicts future stock prices based on past trends. The dataset consists of historical stock market data containing daily closing prices of a selected company, which is preprocessed through normalization and sequence generation before being used for training and testing the RNN model.

#### testset.csv

<img width="755" height="434" alt="image" src="https://github.com/user-attachments/assets/648e9b18-f197-482f-9a7b-553442479ee6" />

#### trainset.csv

<img width="726" height="432" alt="image" src="https://github.com/user-attachments/assets/1998f911-8669-4342-8c33-6b66d09c86c0" />


## DESIGN STEPS

### STEP 1:
Collect historical stock closing price data and perform preprocessing such as normalization and sequence creation

### STEP 2:
Split the dataset into training and testing sets and convert them into PyTorch tensors and DataLoader format.

### STEP 3:
Design an RNN model using input, hidden, and output layers suitable for time-series prediction.

### STEP 4:
Define the loss function (Mean Squared Error) and optimizer (Adam) for training the model.

### STEP 5:
Train the RNN model over multiple epochs while updating weights using backpropagation through time.

### STEP 6:
Evaluate the trained model by plotting training loss and comparing predicted stock prices with actual prices.


## PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

df_train.head()

train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,output_size=1):
      super(RNNModel, self).__init__()
      self.rnn = nn.RNN(input_size, hidden_size, num_layers,batch_first=True)
      self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x):
      out,_=self.rnn(x)
      out=self.fc(out[:,-1,:])
      return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

!pip install torchinfo

from torchinfo import summary
summary(model, input_size=(64, 60, 1))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch =x_batch.to(device),y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    print('Name : R.JAYASREE')
    print('Register Number: 212223040074')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

train_model(model,train_loader,criterion,optimizer)

model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

print('Name: JAYASREE r')
print('Register Number: 212223040074')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')

```
### OUTPUT

### Training Loss Over Epochs Plot

<img width="323" height="432" alt="image" src="https://github.com/user-attachments/assets/d0190bb2-9f97-45e8-833d-a11633e92a81" />

<img width="759" height="623" alt="image" src="https://github.com/user-attachments/assets/a2b71d44-8f5a-460c-a955-71c05567f243" />


### True Stock Price, Predicted Stock Price vs time

<img width="1086" height="720" alt="image" src="https://github.com/user-attachments/assets/a67110c3-7f35-4283-967f-9710b14c30f4" />


### Predictions

<img width="286" height="51" alt="image" src="https://github.com/user-attachments/assets/227df06d-0fcd-4d57-a8f8-323f622617c2" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
