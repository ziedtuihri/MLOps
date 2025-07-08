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

def main():
    mlflow.set_experiment("Productivity_Experiment")
    with mlflow.start_run():
        # Load and preprocess data
        df = load_data("dataProductivityEmployeese.csv")
        df["wip"] = df["wip"].fillna(value=0)
        df = preprocess_data(df)
        print("Data loaded and preprocessed. Shape:", df.shape)
        mlflow.log_param("data_shape", str(df.shape))

        # Split features and target
        X = df.drop(["actual_productivity"], axis=1)
        y = df["actual_productivity"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        X_train_sc, X_test_sc = scale_features(X_train, X_test)

        # Train Random Forest model and log
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
        print(f"Random Forest RMSE: {rmse:.4f}")

        # Train Gradient Boosting model and log
        gb_rmse = train_gradient_boosting(X_train, y_train, X_test, y_test)
        mlflow.log_metric("GradientBoosting_RMSE", gb_rmse)
        print(f"Gradient Boosting RMSE: {gb_rmse:.4f}")

        # Clustering
        print("\nClustering Results:")
        kmeans_labels, kmeans_sil = clustering_kmeans(X_train_sc)
        mlflow.log_metric("KMeans_Silhouette", kmeans_sil)
        print(f"KMeans Silhouette Score: {kmeans_sil:.4f}")

        cah_labels, cah_sil = clustering_cah(X_train_sc)
        mlflow.log_metric("CAH_Silhouette", cah_sil)
        print(f"CAH Silhouette Score: {cah_sil:.4f}")

        dbscan_labels, dbscan_sil = clustering_dbscan(X_train_sc)
        if dbscan_sil is not None:
            mlflow.log_metric("DBSCAN_Silhouette", dbscan_sil)
        print(f"DBSCAN Silhouette Score: {dbscan_sil}")

        gmm_labels, gmm_sil = clustering_gmm(X_train_sc)
        mlflow.log_metric("GMM_Silhouette", gmm_sil)
        print(f"GMM Silhouette Score: {gmm_sil:.4f}")

        # Log environment files as artifacts if they exist
        for fname in ["MLmodel", "conda.yml", "python_env.yaml", "requirements.txt"]:
            try:
                mlflow.log_artifact(fname)
            except Exception:
                pass  # Ignore if file does not exist

if __name__ == "__main__":
    main()

