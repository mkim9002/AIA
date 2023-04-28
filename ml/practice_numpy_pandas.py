import pandas as pd
import numpy as np

# Create a dataset
data = np.array([[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15],
                 [16, 17, 18, 19, 20]])

# Create a DataFrame
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Convert DataFrame to numpy array
numpy_array = df.to_numpy()

print("DataFrame converted to numpy array:")
print(numpy_array)

# Convert numpy array to DataFrame
new_df = pd.DataFrame(numpy_array)

print("Numpy array converted to DataFrame:")
print(new_df)

# Convert list to numpy array
my_list = [21, 22, 23, 24, 25]
numpy_array = np.array(my_list)

print("List converted to numpy array:")
print(numpy_array)

# Convert list to DataFrame
new_df = pd.DataFrame(my_list)

print("List converted to DataFrame:")
print(new_df)
