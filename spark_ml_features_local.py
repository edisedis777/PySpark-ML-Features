from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col, expr, rand
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Spark Session for local environment
def init_spark():
    """
    Initialize a local Spark session.
    
    Returns:
        SparkSession: Local Spark session
    """
    return SparkSession.builder \
        .appName("Local Spark ML Features") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

def load_sample_data(spark):
    """
    Load the Iris dataset using sklearn.datasets.
    
    Args:
        spark: SparkSession object
        
    Returns:
        DataFrame: Loaded Iris dataset
    """
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        iris_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                                columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
        # Convert numeric target to string labels
        iris_data['species'] = iris_data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        # Convert to Spark DataFrame
        df = spark.createDataFrame(iris_data)
    except ImportError:
        # Fallback if scikit-learn is not available
        data = [
            (5.1, 3.5, 1.4, 0.2, "setosa"),
            (4.9, 3.0, 1.4, 0.2, "setosa"),
            (7.0, 3.2, 4.7, 1.4, "versicolor"),
            (6.4, 3.2, 4.5, 1.5, "versicolor"),
            (6.3, 3.3, 6.0, 2.5, "virginica"),
            (5.8, 2.7, 5.1, 1.9, "virginica")
        ]
        df = spark.createDataFrame(data, ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
        print("Created a sample Iris dataset (scikit-learn not available)")
    
    return df

def prepare_features(df, feature_cols=None, label_col="species"):
    """
    Prepare features for modeling by assembling them into a vector and indexing string labels.
    
    Args:
        df: DataFrame with feature columns
        feature_cols: List of feature column names (if None, defaults to Iris dataset columns)
        label_col: Name of the label column
        
    Returns:
        DataFrame: DataFrame with assembled features and indexed labels
    """
    if feature_cols is None:
        feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    # Index the string labels
    indexer = StringIndexer(inputCol=label_col, outputCol="label")
    df_indexed = indexer.fit(df).transform(df)
    
    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_features = assembler.transform(df_indexed).select("features", "label")
    return df_features

def validation_curves(df_features, param_name="regParam", param_range=None, 
                     label_col="label", classifier=None, num_folds=3):
    """
    Create validation curves to visualize how a hyperparameter affects model performance.
    
    Args:
        df_features: DataFrame with features and label
        param_name: Name of the parameter to vary
        param_range: Range of parameter values to test
        label_col: Name of the label column
        classifier: ML classifier to use (defaults to LogisticRegression)
        num_folds: Number of cross-validation folds
        
    Returns:
        tuple: (param_range, metrics) for plotting
    """
    if param_range is None:
        param_range = np.logspace(-6, -1, 5)
    
    if classifier is None:
        classifier = LogisticRegression(labelCol=label_col)
    
    # Define parameter grid
    param_grid = ParamGridBuilder().addGrid(getattr(classifier, param_name), param_range).build()
    
    # Set up CrossValidator
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
    cv = CrossValidator(estimator=classifier, estimatorParamMaps=param_grid, 
                        evaluator=evaluator, numFolds=num_folds)
    
    # Fit the model
    cv_model = cv.fit(df_features)
    
    # Extract metrics
    metrics = cv_model.avgMetrics
    
    return param_range, metrics

def plot_validation_curves(param_range, metrics, param_name="Regularization Parameter"):
    """
    Plot validation curves.
    
    Args:
        param_range: Range of parameter values
        metrics: Performance metrics for each parameter value
        param_name: Name of the parameter (for x-axis label)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, metrics, 'o-', label="Validation Accuracy")
    plt.xscale('log')
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.title("Validation Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("validation_curve.png")
    print(f"Validation curve saved to 'validation_curve.png'")
    return plt

def probability_prediction(df_features, label_col="label", classifier=None):
    """
    Get probability predictions from a classifier.
    Note: This is a simplified alternative to calibration, which isn't natively supported in Spark MLlib.
    
    Args:
        df_features: DataFrame with features and label
        label_col: Name of the label column
        classifier: ML classifier to use (defaults to LogisticRegression)
        
    Returns:
        DataFrame: Predictions with probabilities
    """
    if classifier is None:
        classifier = LogisticRegression(labelCol=label_col, probabilityCol="probability")
    
    # Train the model
    model = classifier.fit(df_features)
    
    # Get predictions with probabilities
    predictions = model.transform(df_features)
    
    return predictions.select(label_col, "probability", "prediction")

def robust_scaling(df, columns=None, quantile_error=0.05):
    """
    Apply robust scaling to features using median and IQR.
    
    Args:
        df: DataFrame with features
        columns: List of columns to scale (if None, uses all numeric columns)
        quantile_error: Error tolerance for quantile approximation
        
    Returns:
        DataFrame: DataFrame with scaled features
    """
    if columns is None:
        # This is a simplified approach; in practice, you might want to filter for numeric columns
        columns = df.columns
    
    result_df = df
    
    for col_name in columns:
        # Calculate median and IQR
        quantiles = df.approxQuantile(col_name, [0.25, 0.5, 0.75], quantile_error)
        median = quantiles[1]
        iqr = quantiles[2] - quantiles[0]
        
        # Skip if IQR is too small to avoid division by zero
        if iqr > 1e-10:
            # Apply robust scaling
            result_df = result_df.withColumn(f"{col_name}_scaled", (col(col_name) - median) / iqr)
    
    return result_df

def feature_union(df_features, transform_types=None):
    """
    Combine multiple feature transformations.
    
    Args:
        df_features: DataFrame with 'features' column
        transform_types: List of transformation types to apply
        
    Returns:
        DataFrame: DataFrame with combined features
    """
    if transform_types is None or "scaled" in transform_types:
        # Apply standard scaling
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                               withStd=True, withMean=True)
        df_scaled = scaler.fit(df_features).transform(df_features)
        
        # Combine with original features
        assembler = VectorAssembler(inputCols=["features", "scaled_features"], 
                                   outputCol="combined_features")
        df_combined = assembler.transform(df_scaled)
        return df_combined
    
    return df_features

def feature_dimensionality_reduction(df_features, method="kmeans", k=2):
    """
    Reduce feature dimensionality using clustering or PCA.
    This is an alternative to feature agglomeration.
    
    Args:
        df_features: DataFrame with features
        method: Method to use ('kmeans' or 'pca')
        k: Number of clusters or components
        
    Returns:
        DataFrame: DataFrame with cluster assignments or reduced features
    """
    if method == "kmeans":
        kmeans = KMeans(k=k, featuresCol="features")
        model = kmeans.fit(df_features)
        return model.transform(df_features)
    elif method == "pca":
        from pyspark.ml.feature import PCA
        pca = PCA(k=k, inputCol="features", outputCol="pca_features")
        model = pca.fit(df_features)
        return model.transform(df_features)
    else:
        raise ValueError(f"Unsupported method: {method}")

def add_split_column(df_features, split_condition="random", label_col="label"):
    """
    Add a split column for predefined train-test splits.
    
    Args:
        df_features: DataFrame with features and label
        split_condition: Condition for splitting ('random' or a custom condition)
        label_col: Name of the label column
        
    Returns:
        DataFrame: DataFrame with added split column
    """
    if split_condition == "random":
        # Random 70-30 split
        return df_features.withColumn("is_train", (rand() < 0.7).cast("int"))
    else:
        # Example of a custom condition (e.g., splitting by a specific class)
        return df_features.withColumn("is_train", (col(label_col) != 0).cast("int"))  # Assuming label 0 is 'setosa'

def predefined_split(df_features, split_col="is_train", param_grid=None, 
                    classifier=None, label_col="label"):
    """
    Use a predefined train-test split for validation.
    
    Args:
        df_features: DataFrame with features and label
        split_col: Column indicating train/test split (1 for train, 0 for test)
        param_grid: Parameter grid for model tuning
        classifier: ML classifier to use
        label_col: Name of the label column
        
    Returns:
        tuple: (model, train_df, test_df)
    """
    # Filter train and test data
    train_df = df_features.filter(col(split_col) == 1)
    test_df = df_features.filter(col(split_col) == 0)
    
    if classifier is None:
        classifier = LogisticRegression(labelCol=label_col)
    
    if param_grid is None:
        param_grid = ParamGridBuilder().addGrid(classifier.regParam, [0.1, 0.01]).build()
    
    # Set up evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
    
    # Use TrainValidationSplit with 100% train ratio (since we have a predefined split)
    tvs = TrainValidationSplit(estimator=classifier, estimatorParamMaps=param_grid, 
                              evaluator=evaluator, trainRatio=1.0)
    
    # Fit the model
    model = tvs.fit(train_df)
    
    return model, train_df, test_df

# Example usage
def demo_all_features(spark):
    """
    Demonstrate all six features on the Iris dataset.
    
    Args:
        spark: SparkSession object
    """
    # Load data
    df = load_sample_data(spark)
    print("Loaded Iris dataset:")
    df.show(5)
    
    # Prepare features
    df_features = prepare_features(df)
    print("Prepared features:")
    df_features.show(5)
    
    # 1. Validation Curves
    param_range, metrics = validation_curves(df_features)
    print("Validation curve metrics:", metrics)
    
    # Plot the validation curve
    plot_validation_curves(param_range, metrics)
    
    # 2. Probability Prediction (alternative to calibration)
    predictions = probability_prediction(df_features)
    print("Probability predictions:")
    predictions.show(5)
    
    # 3. Robust Scaling
    df_scaled = robust_scaling(df, columns=["sepal_length", "sepal_width"])
    print("Robust scaling:")
    df_scaled.select("sepal_length", "sepal_length_scaled", "sepal_width", "sepal_width_scaled").show(5)
    
    # 4. Feature Union
    df_combined = feature_union(df_features)
    print("Feature union:")
    df_combined.select("features", "scaled_features", "combined_features").show(5, truncate=True)
    
    # 5. Feature Dimensionality Reduction
    df_clustered = feature_dimensionality_reduction(df_features, method="kmeans", k=2)
    print("Feature dimensionality reduction (KMeans):")
    df_clustered.select("features", "prediction").show(5)
    
    # 6. Predefined Split
    df_with_split = add_split_column(df_features, split_condition="custom")
    model, train_df, test_df = predefined_split(df_with_split, split_col="is_train")
    print("Predefined split - Train set count:", train_df.count())
    print("Predefined split - Test set count:", test_df.count())
    
    return "All features demonstrated successfully!"

if __name__ == "__main__":
    # Initialize Spark session
    spark = init_spark()
    print("Spark session initialized.")
    
    # Run all demonstrations
    result = demo_all_features(spark)
    print(result)
    
    # Stop Spark session when done
    spark.stop()
    print("Spark session stopped.")