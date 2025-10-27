import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from kneed import KneeLocator
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score

st.title("🧠 Auto ML App")
st.write("Upload your dataset, train models with optional hyperparameter tuning, and make predictions!")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    target_column = st.selectbox("🎯 Select target column (the column to predict)", df.columns)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical features
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        # Encode target if categorical
        y_encoder = None
        if y.dtype == 'object':
            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y)

        # Fill missing values
        X = X.fillna(X.mean(numeric_only=True))

        # Scale numeric features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        st.success("✅ Data preprocessed successfully!")

        # Step 3: Choose models
        st.write("### ⚙️ Choose ML Model(s):")
        models_selected = st.multiselect(
            "Select model(s) to apply",
            [
                "Logistic Regression",
                "Random Forest",
                "Support Vector Machine",
                "K-Nearest Neighbors",
                "Decision Tree",
                "Gradient Boosting"
            ]
        )

        auto_best = st.checkbox("🔍 Automatically select the best model")
        tune_params = st.checkbox("🔧 Enable Hyperparameter Tuning (may take longer)")

        # Hyperparameter grids
        param_grids = {
            "Logistic Regression": {"C": [0.1, 1, 10], "solver": ["lbfgs"], "max_iter": [500, 1000]},
            "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
            "Support Vector Machine": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
            "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
            "Decision Tree": {"max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
            "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
        }

        # Step 4: Train models
        if st.button("🚀 Train Model(s)"):
            # If auto_best is checked but no models selected, select all
            if auto_best and not models_selected:
                models_selected = [
                    "Logistic Regression", "Random Forest", "Support Vector Machine",
                    "K-Nearest Neighbors", "Decision Tree", "Gradient Boosting"
                ]

            if not models_selected:
                st.error("❌ No model selected. Please select at least one model or enable auto-best.")
            else:
                results = {}
                trained_models = {}

                for model_name in models_selected:
                    st.write(f"Training {model_name}...")
                    # Initialize model
                    if model_name == "Logistic Regression":
                        model = LogisticRegression()
                    elif model_name == "Random Forest":
                        model = RandomForestClassifier()
                    elif model_name == "Support Vector Machine":
                        model = SVC()
                    elif model_name == "K-Nearest Neighbors":
                        model = KNeighborsClassifier()
                    elif model_name == "Decision Tree":
                        model = DecisionTreeClassifier()
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingClassifier()
                    else:
                        continue

                    # Hyperparameter tuning
                    if tune_params:
                        grid = GridSearchCV(model, param_grids[model_name], cv=3, scoring='accuracy')
                        grid.fit(X_train, y_train)
                        best_model = grid.best_estimator_
                    else:
                        model.fit(X_train, y_train)
                        best_model = model

                    trained_models[model_name] = best_model
                    acc = accuracy_score(y_test, best_model.predict(X_test))
                    results[model_name] = acc

                # Ensure results is not empty before selecting best
                if results:
                    best_model_name = max(results, key=results.get)
                    best_model = trained_models[best_model_name]

                    if auto_best:
                        st.success(f"🏆 Best Model: {best_model_name} (Accuracy: {results[best_model_name]:.2f})")
                    else:
                        st.write("### 📊 Model Accuracies:")
                        st.write(results)

                    # Save best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(best_model, f)
                    st.success("💾 Model saved ")

                    # Store in session state
                    st.session_state["best_model"] = best_model
                    st.session_state["features"] = list(X.columns)
                    st.session_state["scaler"] = scaler
                    st.session_state["y_encoder"] = y_encoder
                else:
                    st.error("❌ Training failed. Please check your dataset and try again.")

# Step 5: Prediction section
st.write("---")
st.header("🔮 Predict with Trained or Saved Model")

model_loaded = None

# Load from session or file
if "best_model" in st.session_state:
    model_loaded = st.session_state["best_model"]
    features = st.session_state["features"]
    scaler = st.session_state["scaler"]
    y_encoder = st.session_state["y_encoder"]
    st.info("Using model trained in this session.")
else:
    if st.button("📂 Load Saved Model"):
        try:
            with open('best_model.pkl', 'rb') as f:
                model_loaded = pickle.load(f)
            st.success("Model loaded successfully!")
        except FileNotFoundError:
            st.error("No saved model found. Train and save one first.")

# Prediction form
if model_loaded is not None:
    st.subheader("🧾 Enter new values for prediction")

    if "features" in st.session_state:
        inputs = []
        for col in st.session_state["features"]:
            val = st.number_input(f"Value for {col}", value=0.0)
            inputs.append(val)

        if st.button("Predict"):
            input_array = np.array(inputs).reshape(1, -1)
            scaled_input = st.session_state["scaler"].transform(input_array)
            prediction = model_loaded.predict(scaled_input)

            # Decode class label if encoder exists
            if "y_encoder" in st.session_state and st.session_state["y_encoder"] is not None:
                decoded_pred = st.session_state["y_encoder"].inverse_transform(prediction)[0]
                st.success(f"🎯 Predicted Class: {decoded_pred}")
            else:
                st.success(f"🎯 Predicted Output: {prediction[0]}")


st.write("---")
st.header("🌀 Clustering with Automatic Elbow Detection")

if uploaded_file is not None:
    st.write("### Select Clustering Algorithm:")
    clustering_algo = st.selectbox(
        "Choose clustering algorithm",
        ["K-Means", "Agglomerative Clustering", "DBSCAN"]
    )

    if clustering_algo:
        X_clustering = X_scaled.copy()  # Use preprocessed features

        if clustering_algo in ["K-Means", "Agglomerative Clustering"]:
            use_elbow = st.checkbox("🔹 Automatically find optimal number of clusters (Elbow Method)")

        if st.button("Run Clustering"):
            if clustering_algo == "K-Means":
                if use_elbow:
                    st.write("Calculating Elbow Curve...")
                    distortions = []
                    K = range(2, 11)
                    for k in K:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(X_clustering)
                        distortions.append(kmeans.inertia_)

                    # Detect the elbow point
                    kneedle = KneeLocator(K, distortions, curve='convex', direction='decreasing')
                    optimal_k = kneedle.knee
                    st.success(f"🏆 Suggested optimal number of clusters: {optimal_k}")

                    # Plot elbow curve and mark the knee
                    plt.figure()
                    plt.plot(K, distortions, 'bx-')
                    plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='red')
                    plt.xlabel('Number of clusters (k)')
                    plt.ylabel('Inertia')
                    plt.title('Elbow Method')
                    st.pyplot(plt)
                else:
                    optimal_k = st.number_input("Number of clusters", min_value=2, max_value=20, value=3, step=1)

                model_cluster = KMeans(n_clusters=optimal_k, random_state=42)

            elif clustering_algo == "Agglomerative Clustering":
                n_clusters = optimal_k if use_elbow and 'optimal_k' in locals() else 3
                model_cluster = AgglomerativeClustering(n_clusters=n_clusters)

            elif clustering_algo == "DBSCAN":
                model_cluster = DBSCAN()

            cluster_labels = model_cluster.fit_predict(X_clustering)
            df_clusters = df.copy()
            df_clusters["Cluster"] = cluster_labels
            st.write("### Cluster Assignments:")
            st.dataframe(df_clusters)

            # Silhouette Score
            try:
                if len(set(cluster_labels)) > 1:
                    score = silhouette_score(X_clustering, cluster_labels)
                    st.success(f"✅ Silhouette Score: {score:.3f}")
                else:
                    st.warning("Silhouette Score cannot be calculated for 1 cluster")
            except Exception as e:
                st.warning(f"Could not calculate silhouette score: {e}")

            # Save clustering model
            with open(f"{clustering_algo}_model.pkl", 'wb') as f:
                pickle.dump(model_cluster, f)
            st.success(f"💾 {clustering_algo} model saved")
