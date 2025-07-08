import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["wip"].fillna(value=0, inplace=True)
    return df

def preprocess_data(df):
    df = df.sort_values(by="date").reset_index(drop=True)
    df["department"] = df.apply(
        lambda row: (
            "finishing"
            if row["wip"] == 0
            else (
                "sewing"
                if pd.isna(row["department"]) and row["wip"] > 0
                else row["department"]
            )
        ),
        axis=1,
    )
    df = df.dropna(subset=["date"])
    for i in range(1, len(df) - 1):
        if pd.isna(df.loc[i, "quarter"]):
            previous_quarter = df.loc[i - 1, "quarter"]
            next_quarter = df.loc[i + 1, "quarter"]
            previous_date = df.loc[i - 1, "date"]
            next_date = df.loc[i + 1, "date"]
            if (
                previous_quarter == next_quarter
                and previous_date.month == next_date.month
                and previous_date.day == next_date.day
            ):
                df.loc[i, "quarter"] = previous_quarter
    df = df.dropna(subset=["quarter"])
    for i in range(1, len(df) - 1):
        if pd.isna(df.iloc[i]["day"]):
            previous_date = df.iloc[i - 1]["date"]
            next_date = df.iloc[i + 1]["date"]
            if previous_date.date() == next_date.date():
                df.iloc[i, df.columns.get_loc("day")] = df.iloc[i - 1]["day"]
    df = df.dropna(subset=["day"])

    def fill_idle_columns(row):
        if pd.isna(row["idle_time"]) and pd.isna(row["idle_men"]):
            row["idle_time"] = 0
            row["idle_men"] = 0
        if pd.isna(row["idle_time"]):
            if row["idle_men"] == 0:
                row["idle_time"] = 0
        if pd.isna(row["idle_men"]):
            if row["idle_time"] == 0:
                row["idle_men"] = 0
        return row

    df = df.apply(fill_idle_columns, axis=1)
    df["incentive"] = df["incentive"].fillna(0)
    df["over_time"] = df["over_time"].fillna(0)
    df = df.dropna(subset=["no_of_workers", "team"])

    def set_no_of_style_change(row):
        if pd.isna(row["no_of_style_change"]) and pd.notna(row["smv"]):
            if row["smv"] == 11.41:
                return 2
            elif row["smv"] == 30.1:
                return 1
        return row["no_of_style_change"]

    df["no_of_style_change"] = df.apply(set_no_of_style_change, axis=1)
    df = df.dropna(
        subset=[
            "no_of_style_change",
            "smv",
            "actual_productivity",
            "targeted_productivity",
        ]
    )
    df["no_of_workers"] = df["no_of_workers"].apply(lambda x: int(x))
    df["actual_productivity"] = pd.to_numeric(
        df["actual_productivity"], errors="coerce"
    )
    df["quarter"] = df["quarter"].astype(str).str.replace("Quarter", "")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce")
    df["department"] = df["department"].str.replace("sweing", "sewing")
    df["department"] = df["department"].str.replace("finishing ", "finishing")
    df["department"] = df["department"].replace({"sewing": 0, "finishing": 1})
    day_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Saturday": 4,
        "Sunday": 5,
    }
    df["day"] = df["day"].replace(day_map)
    for col in df.columns:
        if df[col].dtype == "object" and col != "actual_productivity":
            print(f"Warning: Dropping non-numeric column: {col}")
            df = df.drop(columns=[col])
    df = df.drop(columns=["date"], errors="ignore")
    return df

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    gbr = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.01, max_depth=5, random_state=0
    )
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def clustering_kmeans(X_scaled, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    return labels, sil

def clustering_cah(X_scaled, n_clusters=2):
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters, metric="euclidean", linkage="ward"
    )
    labels = clusterer.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    return labels, sil

def clustering_dbscan(X_scaled, eps=0.7, min_samples=10):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    if len(set(labels)) > 1:
        sil = silhouette_score(X_scaled, labels)
    else:
        sil = None
    return labels, sil

def clustering_gmm(X_scaled, n_components=2):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    return labels, sil

def scale_features(X_train, X_test):
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    return X_train_sc, X_test_sc

