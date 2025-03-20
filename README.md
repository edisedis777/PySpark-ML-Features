# PySpark ML Features

A PySpark implementation of 6 lesser-known Scikit-Learn features optimized for Azure Databricks. This project translates powerful machine learning techniques from Scikit-Learn into PySpark's distributed computing framework, allowing you to apply these techniques to large-scale datasets in a cloud environment.

## 🚀 Features

This library implements PySpark equivalents of six powerful Scikit-Learn features:

1. **Validation Curves**: Visualize model performance across hyperparameter values
2. **Probability Prediction**: Alternative to calibration for classifier probability outputs
3. **Robust Scaling**: Scale features using median and IQR to handle outliers
4. **Feature Union**: Combine multiple feature transformations into one feature set
5. **Feature Dimensionality Reduction**: Alternative to feature agglomeration using clustering
6. **Predefined Split**: Use custom train-test splits for validation

## 📋 Requirements

- PySpark 3.0+
- Azure Databricks Runtime 7.0+ (DBR with ML)
- Python 3.6+
- Matplotlib (for visualization)

## 🔧 Installation

1. Upload the `spark_ml_features.py` file to your Databricks workspace or DBFS
2. Import it in your notebook:

```python
# Import the entire module
from spark_ml_features import *

# Or import specific functions
from spark_ml_features import validation_curves, robust_scaling
```

## 📖 Usage

### Demo All Features

Run the demo function to see all features in action:

```python
demo_all_features(spark)
```

### Individual Features

#### 1. Validation Curves

```python
df = load_sample_data(spark)
df_features = prepare_features(df)

# Generate validation curves
param_range, metrics = validation_curves(
    df_features, 
    param_name="regParam", 
    param_range=np.logspace(-6, -1, 5),
    num_folds=3
)

# Plot the results
plot = plot_validation_curves(param_range, metrics)
display(plot)
```

#### 2. Probability Prediction

```python
# Get probability predictions
predictions = probability_prediction(df_features)
display(predictions)
```

#### 3. Robust Scaling

```python
# Scale features using median and IQR
df_scaled = robust_scaling(df, columns=["sepal_length", "sepal_width"])
display(df_scaled)
```

#### 4. Feature Union

```python
# Combine multiple feature transformations
df_combined = feature_union(df_features)
display(df_combined)
```

#### 5. Feature Dimensionality Reduction

```python
# Reduce dimensions using KMeans
df_clustered = feature_dimensionality_reduction(df_features, method="kmeans", k=2)
display(df_clustered)

# Or use PCA
df_pca = feature_dimensionality_reduction(df_features, method="pca", k=2)
display(df_pca)
```

#### 6. Predefined Split

```python
# Add a split column
df_with_split = add_split_column(df_features, split_condition="custom")

# Use predefined split for validation
model, train_df, test_df = predefined_split(df_with_split, split_col="is_train")
```

## 🧰 Helper Functions

The library includes several helper functions to simplify common tasks:

- **`load_sample_data(spark)`**: Load the Iris dataset from Databricks sample data
- **`prepare_features(df, feature_cols, label_col)`**: Prepare feature vectors for modeling
- **`plot_validation_curves(param_range, metrics)`**: Visualize validation curves
- **`add_split_column(df_features, split_condition)`**: Add a column for predefined splits

## 🔍 Detailed Examples

### Validation Curves Example

```python
# Import necessary functions
from spark_ml_features import load_sample_data, prepare_features, validation_curves, plot_validation_curves

# Load data
df = load_sample_data(spark)
df_features = prepare_features(df)

# Define hyperparameter range
import numpy as np
param_range = np.logspace(-6, -1, 5)

# Generate validation curves
param_range, metrics = validation_curves(
    df_features, 
    param_name="regParam", 
    param_range=param_range,
    label_col="species", 
    num_folds=3
)

# Plot the results
plot = plot_validation_curves(param_range, metrics, param_name="Regularization Parameter")
display(plot)
```

### Robust Scaling Example

```python
# Import necessary functions
from spark_ml_features import load_sample_data, robust_scaling

# Load data
df = load_sample_data(spark)

# Apply robust scaling to specific columns
df_scaled = robust_scaling(df, columns=["sepal_length", "sepal_width"], quantile_error=0.01)

# Display the results
display(df_scaled.select("sepal_length", "sepal_length_scaled", "sepal_width", "sepal_width_scaled"))
```

## 🚢 Real-world Usage

For real-world data in Azure Databricks:

```python
# Load data from Azure Blob Storage
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("wasbs://<container>@<account>.blob.core.windows.net/<path>")

# Prepare features
df_features = prepare_features(df, feature_cols=["col1", "col2", "col3"], label_col="target")

# Apply the techniques
df_scaled = robust_scaling(df, columns=["col1", "col2"])
df_clustered = feature_dimensionality_reduction(df_features, method="kmeans", k=3)
```

## 📝 Notes

- This implementation is designed specifically for Azure Databricks environments
- Some features (like validation curves) might require adaptation for extremely large datasets
- The implementation prioritizes scalability and distributed computing over exact equivalence to Scikit-Learn

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 Credits

This project is inspired by Jason Brownlee's article: [6 Lesser-Known Scikit-Learn Features That Will Save You Time](https://machinelearningmastery.com/6-lesser-known-scikit-learn-features-that-will-save-you-time/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
