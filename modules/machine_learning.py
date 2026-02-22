# Import necessary libraries
import json
import joblib

import pandas as pd
import streamlit as st
import numpy as np

# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Custom classes 
from . import utils
from . import modelUtils
import os

def app():
    """Enhanced machine learning application with preprocessing, visualizations, and better UX."""
    
    # Load the data 
    if 'main_data.csv' not in os.listdir('data'):
        st.warning("⚠️ **No data found!**")
        st.info("📂 Please upload your dataset through the **Upload Data** page first.")
        return
    
    data = pd.read_csv('data/main_data.csv')
    col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')
    
    # ============================================
    # HEADER SECTION
    # ============================================
    st.markdown("## 🤖 Machine Learning")
    st.caption("Train predictive models with automated preprocessing and visualization")
    st.info("""
    📋 **Workflow:** Data Preprocessing → Variable Selection → Problem Type → Train/Test Split → Model Training → Evaluation
    """)
    
    # ============================================
    # SECTION 1: ML PREPROCESSING
    # ============================================
    st.markdown("---")
    st.markdown("### 🔧 Step 1: ML Data Preprocessing")
    st.write("Prepare your data for machine learning by handling useless columns and encoding categorical variables.")
    
    with st.expander("⚙️ **Configure Preprocessing Options**", expanded=True):
        # Identify useless columns
        useless_cols = utils.identify_useless_columns(data, threshold_nulls=0.8, threshold_unique=0.95)
        
        if useless_cols:
            st.warning(f"⚠️ Found {len(useless_cols)} potentially useless columns:")
            useless_df = pd.DataFrame([
                {'Column': item['column'], 'Issues': ', '.join(item['reasons'])}
                for item in useless_cols
            ])
            st.dataframe(useless_df, hide_index=True, use_container_width=True)
            
            remove_useless = st.checkbox(
                "Remove these columns before training",
                value=True,
                help="Removes columns with high nulls, all unique values, or zero variance"
            )
        else:
            st.success("✓ No obviously useless columns detected")
            remove_useless = False
        
        # Show info about categorical encoding
        object_cols = [col for col in data.columns if data[col].dtype == 'object']
        if object_cols:
            st.info(f"ℹ️ **Categorical Encoding:** {len(object_cols)} text columns (Country, Region, etc.) will be automatically encoded to numbers during training using One-Hot Encoding for features and Label Encoding for target variable.")
    
    # Apply preprocessing if checkbox is selected
    data_processed = data.copy()
    preprocessing_log = []
    
    if remove_useless and useless_cols:
        cols_to_remove = [item['column'] for item in useless_cols]
        data_processed = data_processed.drop(columns=cols_to_remove, errors='ignore')
        preprocessing_log.append(f"Removed {len(cols_to_remove)} useless columns")
    
    # ============================================
    # SECTION 2: VARIABLE SELECTION
    # ============================================
    st.markdown("---")
    st.markdown("### 📊 Step 2: Variable Selection")
    st.write("""
    - **Y variable (Target):** The value you want to predict
    - **X variables (Features):** The data used to make predictions
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        y_var = st.radio(
            "Select Target Variable (Y)",
            options=data_processed.columns,
            help="The variable you want to predict"
        )
    
    with col2:
        X_var = st.multiselect(
            "Select Feature Variables (X)",
            options=data_processed.columns,
            help="Variables used to predict the target"
        )
    
    # Validation
    if len(X_var) == 0:
        st.error("⚠️ Please select at least one X variable.")
        return
    
    if y_var in X_var:
        st.error("⚠️ Target variable (Y) cannot be in feature variables (X).")
        return
    
    # Preview selected data
    with st.expander("👁️ **Preview Selected Data**"):
        preview_cols = [y_var] + X_var
        st.write(f"Showing first 10 rows of selected variables:")
        st.dataframe(data_processed[preview_cols].head(10), use_container_width=True)
        
        # Show basic stats
        st.write("**Quick Statistics:**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Target Variable", y_var)
            st.write(f"Type: {data_processed[y_var].dtype}")
            st.write(f"Unique values: {data_processed[y_var].nunique()}")
        with col_b:
            st.metric("Feature Variables", len(X_var))
            numeric_features = sum(1 for col in X_var if utils.isNumerical(data_processed[col]))
            st.write(f"Numeric: {numeric_features}, Categorical: {len(X_var) - numeric_features}")
    
    # ============================================
    # SECTION 3: PREDICTION TYPE SELECTION
    # ============================================
    st.markdown("---")
    st.markdown("### 🎯 Step 3: Select Prediction Type")
    
    pred_type = st.radio(
        "What type of prediction?",
        options=["Regression", "Classification"],
        help="""
        - **Regression:** Predict continuous values (e.g., price, temperature, sales)
        - **Classification:** Predict categories (e.g., spam/not spam, disease type, customer segment)
        """
    )
    
    # Check if prediction type makes sense
    y_unique = data_processed[y_var].nunique()
    if pred_type == "Classification" and y_unique > 20:
        st.warning(f"⚠️ Target variable has {y_unique} unique values. Classification works best with fewer categories.")
    elif pred_type == "Regression" and y_unique < 10:
        st.warning(f"⚠️ Target variable has only {y_unique} unique values. Consider using Classification instead.")
    
    # ============================================
    # SECTION 4: DATA PREPARATION & ENCODING
    # ============================================
    st.markdown("---")
    st.markdown("### 🔄 Step 4: Train/Test Split")
    st.write("""
    Split data into training and testing sets. The model learns from training data and is evaluated on testing data to assess its real-world performance.
    """)
    
    size = st.slider(
        "Training Data Percentage",
        min_value=0.1,
        max_value=0.9,
        step=0.1,
        value=0.8,
        help="Percentage of data used for training. The rest is used for testing."
    )
    
    # Prepare data
    X = data_processed[X_var].copy()
    y = data_processed[y_var].copy()
    
    # Encoding preprocessing summary
    encoding_log = []
    
    # Encode X variables (One-Hot Encoding)
    X_original_cols = X.columns.tolist()
    X_encoded = pd.get_dummies(X, drop_first=False)
    if X_encoded.shape[1] != X.shape[1]:
        encoding_log.append(f"One-hot encoded {X.shape[1]} features → {X_encoded.shape[1]} encoded features")
    
    # Encode Y variable if needed
    le = None
    class_mapping = None
    if not utils.isNumerical(y):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
        encoding_log.append(f"Label encoded target variable: {len(le.classes_)} classes")
    else:
        y_encoded = y
    
    # Show encoding summary
    if encoding_log:
        st.info("🔄 **Encoding Applied:**\n" + "\n".join(f"- {log}" for log in encoding_log))
        
        if class_mapping:
            with st.expander("📋 **View Class Mappings**"):
                st.write("Target variable encoding:")
                for cls, idx in class_mapping.items():
                    st.write(f"- **{cls}** → {idx}")
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, train_size=size, random_state=42
    )
    
    # Display split info
    col_split1, col_split2, col_split3 = st.columns(3)
    col_split1.metric("Total Samples", len(data_processed))
    col_split2.metric("Training Samples", X_train.shape[0])
    col_split3.metric("Testing Samples", X_test.shape[0])
    
    # Save model parameters
    params = {
        'X': X_var,
        'y': y_var,
        'pred_type': pred_type,
        'preprocessing': {
            'removed_useless': remove_useless,
            'encoding_applied': len(encoding_log) > 0
        }
    }
    
    with open('data/metadata/model_params.json', 'w') as json_file:
        json.dump(params, json_file)
    
    # ============================================
    # SECTION 5: MODEL TRAINING
    # ============================================
    st.markdown("---")
    st.markdown("### 🚀 Step 5: Train Models")
    st.write("Training multiple models and comparing their performance...")
    
    if st.button("🎯 Train Models", type="primary"):
        
        if pred_type == "Regression":
            st.markdown("#### Regression Models")
            st.write("""
            - **Linear Regression:** Simple, assumes linear relationship
            - **Decision Tree:** Handles non-linear patterns, can overfit
            """)
            
            model_results = []
            trained_models = []
            
            # Linear Regression
            with st.spinner("Training Linear Regression..."):
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                lr_score = lr_model.score(X_test, y_test)
                lr_pred = lr_model.predict(X_test)
                model_results.append({
                    'Model': 'Linear Regression',
                    'R² Score': lr_score,
                    'model_obj': lr_model,
                    'predictions': lr_pred
                })
                trained_models.append(('Linear Regression', lr_model, lr_pred))
            
            # Decision Tree
            with st.spinner("Training Decision Tree..."):
                dt_model = DecisionTreeRegressor(random_state=42)
                dt_model.fit(X_train, y_train)
                dt_score = dt_model.score(X_test, y_test)
                dt_pred = dt_model.predict(X_test)
                model_results.append({
                    'Model': 'Decision Tree',
                    'R² Score': dt_score,
                    'model_obj': dt_model,
                    'predictions': dt_pred
                })
                trained_models.append(('Decision Tree', dt_model, dt_pred))
            
            # Display results
            st.success("✓ Training completed!")
            
            # Results table
            results_df = pd.DataFrame(model_results)[['Model', 'R² Score']].sort_values(by='R² Score', ascending=False)
            st.dataframe(results_df.style.format({'R² Score': '{:.4f}'}), hide_index=True, use_container_width=True)
            
            # Interpretation
            best_model = results_df.iloc[0]
            st.info(f"🏆 **Best Model:** {best_model['Model']} with R² Score of {best_model['R² Score']:.4f}\n\n"
                   f"💡 R² Score measures how well the model fits the data (1.0 is perfect, negative is very poor).")
            
            # Save best model
            if dt_score > lr_score:
                joblib.dump(dt_model, 'data/metadata/model_reg.sav')
                best_model_obj = dt_model
                best_pred = dt_pred
                best_name = 'Decision Tree'
            else:
                joblib.dump(lr_model, 'data/metadata/model_reg.sav')
                best_model_obj = lr_model
                best_pred = lr_pred
                best_name = 'Linear Regression'
            
            # Visualizations
            st.markdown("---")
            st.markdown("### 📈 Model Performance Visualization")
            
            # Actual vs Predicted
            st.markdown("#### Actual vs Predicted Values")
            st.write("Points closer to the red line indicate better predictions.")
            fig = modelUtils.plot_regression_results(y_test, best_pred, best_name)
            st.pyplot(fig)
            
            # Feature Importance
            st.markdown("#### Feature Importance")
            st.write("Which features contribute most to predictions?")
            fig_imp = modelUtils.plot_feature_importance(best_model_obj, X_encoded.columns, best_name, top_n=10)
            if fig_imp:
                st.pyplot(fig_imp)
        
        elif pred_type == "Classification":
            st.markdown("#### Classification Models")
            st.write("""
            - **Logistic Regression:** Simple, fast, works well for binary classification
            - **Decision Tree:** Handles complex patterns, interpretable
            """)
            
            model_results = []
            trained_models = []
            
            # Logistic Regression
            with st.spinner("Training Logistic Regression..."):
                lc_model = LogisticRegression(max_iter=1000, random_state=42)
                lc_model.fit(X_train, y_train)
                lc_score = lc_model.score(X_test, y_test)
                lc_pred = lc_model.predict(X_test)
                model_results.append({
                    'Model': 'Logistic Regression',
                    'Accuracy': lc_score,
                    'model_obj': lc_model,
                    'predictions': lc_pred
                })
                trained_models.append(('Logistic Regression', lc_model, lc_pred))
            
            # Decision Tree
            with st.spinner("Training Decision Tree..."):
                dtc_model = DecisionTreeClassifier(random_state=42)
                dtc_model.fit(X_train, y_train)
                dtc_score = dtc_model.score(X_test, y_test)
                dtc_pred = dtc_model.predict(X_test)
                model_results.append({
                    'Model': 'Decision Tree',
                    'Accuracy': dtc_score,
                    'model_obj': dtc_model,
                    'predictions': dtc_pred
                })
                trained_models.append(('Decision Tree', dtc_model, dtc_pred))
            
            # Display results
            st.success("✓ Training completed!")
            
            # Results table
            results_df = pd.DataFrame(model_results)[['Model', 'Accuracy']].sort_values(by='Accuracy', ascending=False)
            st.dataframe(results_df.style.format({'Accuracy': '{:.4f}'}), hide_index=True, use_container_width=True)
            
            # Interpretation
            best_model = results_df.iloc[0]
            st.info(f"🏆 **Best Model:** {best_model['Model']} with Accuracy of {best_model['Accuracy']:.4f}\n\n"
                   f"💡 Accuracy shows the percentage of correct predictions.")
            
            # Save best model
            if dtc_score > lc_score:
                joblib.dump(dtc_model, 'data/metadata/model_classification.sav')
                best_model_obj = dtc_model
                best_pred = dtc_pred
                best_name = 'Decision Tree'
            else:
                joblib.dump(lc_model, 'data/metadata/model_classification.sav')
                best_model_obj = lc_model
                best_pred = lc_pred
                best_name = 'Logistic Regression'
            
            # Visualizations
            st.markdown("---")
            st.markdown("### 📈 Model Performance Visualization")
            
            # Confusion Matrix
            st.markdown("#### Confusion Matrix")
            st.write("Shows how many predictions were correct vs incorrect for each class.")
            class_names = list(class_mapping.keys()) if class_mapping else None
            fig = modelUtils.plot_confusion_matrix(y_test, best_pred, best_name, class_names)
            st.pyplot(fig)
            
            # Feature Importance
            st.markdown("#### Feature Importance")
            st.write("Which features are most important for classification?")
            fig_imp = modelUtils.plot_feature_importance(best_model_obj, X_encoded.columns, best_name, top_n=10)
            if fig_imp:
                st.pyplot(fig_imp)
