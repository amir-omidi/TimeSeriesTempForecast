# TempPredictMLP

*Forecast daily temperatures with TempPredictMLP, a Python project utilizing a Multi-Layer Perceptron (MLP) neural network. The project covers data preparation, normalization, and model training for accurate time series predictions.*

## Overview

This project involves using a neural network to predict temperature values in a time series. The dataset used for this project is daily mean temperatures, loaded from an Excel file. The process involves data preparation, normalization, creating input-output pairs, and training a neural network for prediction.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

**Data Preparation**: Load the dataset from the 'daily-mean-temperatures.xlsx' file.

```python
data = pd.read_excel('daily-mean-temperatures.xlsx')
```

**Data Normalization**: Normalize the data to ensure consistent training.

```python
mn = np.min(data)
mx = np.max(data)
data_norm = (data - mn) / (mx - mn)
```


##Author

**Amirhossein Omidi

##Contact

**65mirhossein@gmail.com
