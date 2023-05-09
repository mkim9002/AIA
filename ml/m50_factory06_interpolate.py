# Import libraries
import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# Define paths and filenames
path = 'c:/study/_data/AIFac_pollution/'
save_path = 'c:/study/_save/AIFac_pollution/'
submission = pd.read_csv(path + 'answer_sample.csv')
train_files = glob.glob(path + "TRAIN/*.csv")
test_input_files = glob.glob(path + 'test_input/*.csv') 

# Load and concatenate train data
li = []
for filename in train_files:
    df = pd.read_csv(filename, index_col=None, header=0,
                     encoding='utf-8-sig')
    li.append(df)
train_dataset = pd.concat(li, axis=0,
                          ignore_index=True)

# Load and concatenate test data
li = []
for filename in test_input_files:
    df = pd.read_csv(filename, index_col=None, header=0,
                     encoding='utf-8-sig')
    li.append(df)
test_input_dataset = pd.concat(li, axis=0,
                          ignore_index=True)

# Encode location column
le = LabelEncoder()
train_dataset['locate'] = le.fit_transform(train_dataset['측정소'])
test_input_dataset['locate'] = le.transform(test_input_dataset['측정소'])

# Drop unnecessary columns
train_dataset = train_dataset.drop(['측정소'], axis=1)
test_input_dataset = test_input_dataset.drop(['측정소'], axis=1)

# Extract month and hour from date column
train_dataset['month'] = train_dataset['일시'].str[:2].astype(int)
train_dataset['hour'] = train_dataset['일시'].str[6:8].astype(int)
train_dataset = train_dataset.drop(['일시'], axis=1)

test_input_dataset['month'] = test_input_dataset['일시'].str[:2].astype(int)
test_input_dataset['hour'] = test_input_dataset['일시'].str[6:8].astype(int)
test_input_dataset = test_input_dataset.drop(['일시'], axis=1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_dataset.drop(['PM2.5'], axis=1), train_dataset['PM2.5'], test_size=0.2, random_state=42)

# Convert dataframes to tensors
X_train = torch.tensor(X_train.values, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.float).unsqueeze(1)
X_val = torch.tensor(X_val.values, dtype=torch.float)
y_val = torch.tensor(y_val.values, dtype=torch.float).unsqueeze(1)
X_test = torch.tensor(test_input_dataset.values, dtype=torch.float)

# Add a dummy sequence dimension to the input tensors
X_train = X_train.unsqueeze(1) # shape: (batch_size, 1, input_size)
X_val = X_val.unsqueeze(1) # shape: (batch_size, 1, input_size)
X_test = X_test.unsqueeze(1) # shape: (batch_size, 1, input_size)

# Define LSTM model parameters
input_size = X_train.shape[2] # number of features
hidden_size = 64 # number of hidden units
num_layers = 2 # number of LSTM layers
output_size = 1 # number of output units
dropout = 0.2 # dropout probability
bidirectional = False # whether to use bidirectional LSTM

# Define LSTM model class 
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, bidirectional):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (bidirectional + 1), output_size) # multiply by 2 if bidirectional
    
    def forward(self, x):
        # Initialize hidden and cell states 
        h0 = torch.zeros(self.num_layers * (bidirectional + 1), x.size(0), self.hidden_size) # multiply by 2 if bidirectional
        c0 = torch.zeros(self.num_layers * (bidirectional + 1), x.size(0), self.hidden_size) # multiply by 2 if bidirectional
        
        # Forward propagate LSTM
out, _ = self.lstm(x, (h0, c0))  
        
# Decode the hidden state of the last time step
out = self.fc(out[:, -1, :])
return out

# Create an instance of the model
model = LSTM(input_size, hidden_size, num_layers, output_size, dropout, bidirectional)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
num_epochs = 100 # number of epochs to train
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# Predict the PM2.5 values for the test data
y_pred = model(X_test)
y_pred = y_pred.detach().numpy() # convert to numpy array

# Save the predictions to a csv file
submission['PM2.5'] = y_pred
submission.to_csv(save_path + 'submission.csv', index=False)