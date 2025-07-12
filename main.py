import mlflow
import mlflow.sklearn
import joblib
from functionalities import (
    load_data,
    preprocess_data,
    train_gradient_boosting,
    clustering_kmeans,
    clustering_cah,
    clustering_dbscan,
    clustering_gmm,
    scale_features,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from logging_config import setup_logging, get_logger
import logging

# Enable auto-logging
mlflow.autolog()  # <-- Magic happens here!

# Setup logging
logger = setup_logging('mlflow.log', level=logging.INFO)

def main():
    logger.info("Starting MLflow experiment: Productivity_Experiment")
    mlflow.set_experiment("Productivity_Experiment")
    with mlflow.start_run():
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = load_data("dataProductivityEmployeese.csv")
        df["wip"] = df["wip"].fillna(value=0)
        df = preprocess_data(df)
        logger.info(f"Data loaded and preprocessed. Shape: {df.shape}")
        mlflow.log_param("data_shape", str(df.shape))

        # Split features and target
        X = df.drop(["actual_productivity"], axis=1)
        y = df["actual_productivity"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        X_train_sc, X_test_sc = scale_features(X_train, X_test)
        logger.info("Data split completed - Training set: %s, Test set: %s", X_train.shape, X_test.shape)

        # Train Random Forest model and log
        logger.info("Training Random Forest model...")
        rf_model = RandomForestRegressor(
            n_estimators=50, random_state=0, min_samples_split=10, max_depth=6
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_rf)
        rmse = np.sqrt(mse)
        mlflow.log_metric("RandomForest_RMSE", rmse)
        mlflow.sklearn.log_model(rf_model, "random_forest_model")
        joblib.dump(rf_model, "model.pkl")
        mlflow.log_artifact("model.pkl")
        logger.info(f"Random Forest RMSE: {rmse:.4f}")

        # Train Gradient Boosting model and log
        logger.info("Training Gradient Boosting model...")
        gb_rmse = train_gradient_boosting(X_train, y_train, X_test, y_test)
        mlflow.log_metric("GradientBoosting_RMSE", gb_rmse)
        logger.info(f"Gradient Boosting RMSE: {gb_rmse:.4f}")

        # Clustering
        logger.info("Starting clustering analysis...")
        print("\nClustering Results:")
        kmeans_labels, kmeans_sil = clustering_kmeans(X_train_sc)
        mlflow.log_metric("KMeans_Silhouette", kmeans_sil)
        logger.info(f"KMeans Silhouette Score: {kmeans_sil:.4f}")
        print(f"KMeans Silhouette Score: {kmeans_sil:.4f}")

        cah_labels, cah_sil = clustering_cah(X_train_sc)
        mlflow.log_metric("CAH_Silhouette", cah_sil)
        logger.info(f"CAH Silhouette Score: {cah_sil:.4f}")
        print(f"CAH Silhouette Score: {cah_sil:.4f}")

        dbscan_labels, dbscan_sil = clustering_dbscan(X_train_sc)
        if dbscan_sil is not None:
            mlflow.log_metric("DBSCAN_Silhouette", dbscan_sil)
            logger.info(f"DBSCAN Silhouette Score: {dbscan_sil}")
        else:
            logger.warning("DBSCAN clustering failed or returned None")
        print(f"DBSCAN Silhouette Score: {dbscan_sil}")

        gmm_labels, gmm_sil = clustering_gmm(X_train_sc)
        mlflow.log_metric("GMM_Silhouette", gmm_sil)
        logger.info(f"GMM Silhouette Score: {gmm_sil:.4f}")
        print(f"GMM Silhouette Score: {gmm_sil:.4f}")

        # Log environment files as artifacts if they exist
        logger.info("Logging environment artifacts...")
        for fname in ["MLmodel", "conda.yml", "python_env.yaml", "requirements.txt"]:
            try:
                mlflow.log_artifact(fname)
                logger.debug(f"Successfully logged artifact: {fname}")
            except Exception as e:
                logger.debug(f"Could not log artifact {fname}: {e}")
                pass  # Ignore if file does not exist

        logger.info("MLflow experiment completed successfully")

if __name__ == "__main__":
    main()
