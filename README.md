# GAN for Generating Synthetic Financial Transactions

This repository contains my first attempt at using Generative Adversarial Networks (GANs) to generate synthetic financial transactions. The aim is to develop a tool that can help in creating datasets for testing and improving fraud detection systems. While the results are not perfect yet, this project marks the beginning of my journey in applying GANs to finance.

## Overview

Generative Adversarial Networks are a fascinating area of machine learning. In this project, I've explored their potential to generate synthetic financial transaction data. The primary goal is to eventually create realistic datasets that include both legitimate and fraudulent transactions, which are crucial for developing robust fraud detection systems.

## Features

- **Generator Network**: Produces synthetic financial transaction data.
- **Discriminator Network**: Evaluates the authenticity of the generated data.
- **Wasserstein GAN with Gradient Penalty (WGAN-GP)**: An advanced GAN architecture used to stabilize training and improve data quality.
- **Data Preprocessing**: Utilizes techniques like MinMax scaling to prepare the data for GAN training.

## Datasets

The dataset used in this project is the **PaySim** dataset, which simulates mobile money transactions based on real-world data. The dataset includes various features such as transaction amount, type, and whether the transaction is flagged as fraud.

- **Source**: [Kaggle - Fraud Detection on PaySim Dataset](https://www.kaggle.com/code/kartik2112/fraud-detection-on-paysim-dataset)
- **Description**: The PaySim dataset contains 6,362,620 transactions with 11 features. Key features include:
  - `step`: Time step in hours
  - `type`: Type of transaction (e.g., CASH-IN, CASH-OUT, TRANSFER)
  - `amount`: Transaction amount
  - `nameOrig`: Customer ID initiating the transaction
  - `oldbalanceOrg`: Initial balance before the transaction
  - `newbalanceOrig`: New balance after the transaction
  - `nameDest`: Customer ID receiving the transaction
  - `oldbalanceDest`: Initial balance of the recipient before the transaction
  - `newbalanceDest`: New balance of the recipient after the transaction
  - `isFraud`: Flag indicating if the transaction is fraudulent
  - `isFlaggedFraud`: Flag indicating if the transaction is flagged as fraud by the system

## Exploratory Data Analysis (EDA)

A high-level summary of the EDA performed:

1. **Data Distribution**:
   - Analyzed the distribution of transaction types and amounts.
   - Identified that fraudulent transactions are significantly less frequent compared to legitimate ones.

2. **Feature Correlations**:
   - Examined correlations between features such as transaction amount and account balances.
   - Found that certain transaction types had higher correlations with fraud.

3. **Fraud Patterns**:
   - Investigated patterns in fraudulent transactions.
   - Observed that certain transaction types (e.g., CASH-OUT, TRANSFER) are more prone to fraud.

4. **Visualization**:
   - Used histograms, scatter plots, and heatmaps to visualize data distributions and correlations.

## Challenges and Learning Experiences

1. **Imbalanced Data**:
   - The dataset is highly imbalanced with a small percentage of fraudulent transactions.
   - This imbalance posed challenges in training the GAN, as the generator tended to produce more legitimate transactions.

2. **Training Stability**:
   - GANs are known for their instability during training.
   - Implemented Wasserstein GAN with Gradient Penalty (WGAN-GP) to stabilize training.

3. **Hyperparameter Tuning**:
   - Faced challenges in tuning hyperparameters such as learning rate, batch size, and gradient penalty coefficient.
   - Adjusted these parameters through trial and error to find a more stable training setup.

4. **Data Preprocessing**:
   - Ensured proper scaling and normalization of data to improve model performance.
   - Addressed issues related to feature scaling to prevent model bias.

## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/GAN-financial-transactions.git
   cd GAN-financial-transactions

## Usage
### Prepare your dataset:
Ensure your dataset is in a CSV file with appropriate columns for financial transactions.
Typical columns might include amount, step, isFraud, and other relevant transaction details.

### Preprocess the data:
Use the provided code to scale and preprocess your data before feeding it to the GAN model.

### Train the GAN model:

Open the Jupyter Notebook GAN_to_generate_synthetic_financial_transactions_for_developing_fraud_detection_systems_.ipynb.
Follow the steps in the notebook to train the generator and discriminator networks.
Adjust hyperparameters as needed.

### Generate synthetic data:
After training, use the generator to produce synthetic financial transactions.
Save and analyze the synthetic data.

## Example
Below is a basic example of how to preprocess your data, train the GAN model, and generate synthetic transactions:

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

### Load your dataset
df = pd.read_csv('your_dataset.csv')

### Preprocess the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df.drop(['isFraud', 'isFlaggedFraud'], axis=1).values)
dataset = TensorDataset(torch.tensor(df_scaled, dtype=torch.float32))

### Train the GAN model
### (Refer to the Jupyter Notebook for detailed training steps)

### Generate synthetic data
num_samples = 1000
synthetic_samples = generate_synthetic_samples(generator, num_samples, input_dim, device)

### Save synthetic data to CSV
synthetic_df = pd.DataFrame(synthetic_samples, columns=df.drop(['isFraud', 'isFlaggedFraud'], axis=1).columns)
synthetic_df.to_csv('synthetic_transactions.csv', index=False)

## Results and Future Work
The current results of the synthetic data generation are not yet optimal, reflecting the challenges and learning curve associated with my first GAN project in finance. I plan to continue improving this model as I gain more experience and knowledge in this area.

## Contributing
Feedback and contributions are welcome! Please feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
Special thanks to the open-source community for the tools and libraries used in this project.
Inspired by ongoing research and applications of GANs in synthetic data generation.

