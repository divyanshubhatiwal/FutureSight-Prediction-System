import streamlit as st
import numpy as np
import pandas as pd
from modules import utils
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def app():
    """Enhanced data visualization and analysis page with comprehensive insights."""
    
    if 'main_data.csv' not in os.listdir('data'):
        st.warning("⚠️ **No data found!**")
        st.info("📂 Please upload your dataset through the **Upload Data** page to begin analysis.")
        st.markdown("")
        st.markdown("#### Quick Start:")
        st.markdown("1. Navigate to **Upload Data** in the sidebar")
        st.markdown("2. Upload your CSV file")
        st.markdown("3. Click **Load Data** to process")
        st.markdown("4. Return here to visualize!")
        return
    
    # Load data
    df = pd.read_csv('data/main_data.csv')
    col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')
    Categorical, Numerical, Object = utils.getColumnTypes(col_metadata)
    
    # ============================================
    # HEADER SECTION
    # ============================================
    st.markdown("## 📈 Data Analysis & Visualization")
    st.caption("Explore your data through interactive visualizations and statistical analysis")
    st.info("""
    💡 **Explore:** Dataset Overview → Univariate Analysis → Bivariate Analysis → Correlation Matrix → Filtered Deep Dive
    """)
    
    # ============================================
    # SECTION 1: DATASET OVERVIEW
    # ============================================
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Numerical Columns", len(Numerical))
    col4.metric("Categorical Columns", len(Categorical) + len(Object))
    
    with st.expander("📋 **View Raw Data**"):
        st.dataframe(df, use_container_width=True)
    
    with st.expander("📐 **Statistical Summary**"):
        st.write("**Numerical Columns:**")
        if Numerical:
            st.dataframe(df[Numerical].describe(), use_container_width=True)
        else:
            st.info("No numerical columns in the dataset")
        
        st.write("**Categorical Columns:**")
        if Categorical or Object:
            cat_cols = Categorical + Object
            cat_summary = []
            for col in cat_cols:
                cat_summary.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Common': df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A',
                    'Frequency': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
                })
            st.dataframe(pd.DataFrame(cat_summary), hide_index=True, use_container_width=True)
        else:
            st.info("No categorical columns in the dataset")
    
    # ============================================
    # SECTION 2: UNIVARIATE ANALYSIS
    # ============================================
    st.markdown("---")
    st.markdown("### 🔍 Univariate Analysis")
    st.caption("Examine individual variables to understand their distribution patterns")
    st.markdown("")
    
    # Tabs for Numerical and Categorical
    tab1, tab2 = st.tabs([" 📊 Numerical Variables", "🏷️ Categorical Variables"])
    
    with tab1:
        if not Numerical:
            st.info("No numerical columns available for analysis.")
        else:
            st.markdown("#### Select a Numerical Variable")
            num_col = st.selectbox("Choose Column", Numerical, key="univar_num")
            
            # Statistics
            col_data = df[num_col].dropna()
            
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Mean", f"{col_data.mean():.2f}")
            col_b.metric("Median", f"{col_data.median():.2f}")
            col_c.metric("Std Dev", f"{col_data.std():.2f}")
            col_d.metric("Range", f"{col_data.max() - col_data.min():.2f}")
            
            # Visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("**Distribution (Histogram)**")
                fig, ax = plt.subplots()
                ax.hist(col_data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                ax.set_xlabel(num_col)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {num_col}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with viz_col2:
                st.markdown("**Box Plot**")
                fig, ax = plt.subplots()
                ax.boxplot(col_data, vert=True)
                ax.set_ylabel(num_col)
                ax.set_title(f'Box Plot of {num_col}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Outlier detection
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                st.warning(f"⚠️ Detected {len(outliers)} potential outliers ({len(outliers)/len(col_data)*100:.1f}% of data)")
    
    with tab2:
        if not (Categorical or Object):
            st.info("No categorical columns available for analysis.")
        else:
            st.markdown("#### Select a Categorical Variable")
            cat_cols = Categorical + Object
            cat_col = st.selectbox("Choose Column", cat_cols, key="univar_cat")
            
            # Value counts
            value_counts = df[cat_col].value_counts()
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Unique Categories", df[cat_col].nunique())
            col_b.metric("Most Common", value_counts.index[0])
            col_c.metric("Frequency", value_counts.iloc[0])
            
            # Visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("**Pie Chart**")
                # Limit to top 10 categories
                top_n = 10
                if len(value_counts) > top_n:
                    sizes = value_counts.head(top_n)
                    other_sum = value_counts.iloc[top_n:].sum()
                    sizes['Other'] = other_sum
                else:
                    sizes = value_counts
                
                fig, ax = plt.subplots()
                colors = plt.cm.Set3(range(len(sizes)))
                explode = [0.1 if i == 0 else 0 for i in range(len(sizes))]
                ax.pie(sizes, labels=sizes.index, autopct='%1.1f%%', startangle=90, 
                       colors=colors, explode=explode, shadow=True)
                ax.set_title(f'Distribution of {cat_col}')
                st.pyplot(fig)
            
            with viz_col2:
                st.markdown("**Bar Chart**")
                fig, ax = plt.subplots()
                top_10 = value_counts.head(10)
                ax.barh(range(len(top_10)), top_10.values, color='steelblue')
                ax.set_yticks(range(len(top_10)))
                ax.set_yticklabels(top_10.index)
                ax.invert_yaxis()
                ax.set_xlabel('Count')
                ax.set_title(f'Top 10 Categories in {cat_col}')
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
    
    # ============================================
    # SECTION 3: BIVARIATE ANALYSIS
    # ============================================
    st.markdown("---")
    st.markdown("### 🔗 Bivariate Analysis")
    st.caption("Explore relationships and patterns between two variables")
    st.markdown("")
    
    analysis_type = st.radio(
        "**Select Analysis Type:**",
        ["📊 Numerical vs Numerical", "🏷️ Categorical vs Numerical", "🏷️ Categorical vs Categorical"],
        horizontal=True,
        help="Choose the type of variables you want to analyze together"
    )
    
    if analysis_type == "📊 Numerical vs Numerical":
        if len(Numerical) < 2:
            st.info("Need at least 2 numerical columns for this analysis.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Select X Variable", Numerical, key="bivar_x")
            with col2:
                y_var = st.selectbox("Select Y Variable", [col for col in Numerical if col != x_var], key="bivar_y")
            
            # Scatter plot
            st.markdown("**Scatter Plot**")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df[x_var], df[y_var], alpha=0.6, edgecolors='k', s=50)
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.set_title(f'{y_var} vs {x_var}')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(df[x_var].dropna(), df[y_var].dropna(), 1)
            p = np.poly1d(z)
            ax.plot(df[x_var], p(df[x_var]), "r--", alpha=0.8, linewidth=2, label='Trend Line')
            ax.legend()
            
            st.pyplot(fig)
            
            # Correlation
            correlation = df[x_var].corr(df[y_var])
            st.metric("Correlation Coefficient", f"{correlation:.3f}")
            
            if abs(correlation) > 0.7:
                st.success("Strong correlation detected!")
            elif abs(correlation) > 0.4:
                st.info("Moderate correlation detected.")
            else:
                st.warning("Weak or no correlation.")
    
    elif analysis_type == "🏷️ Categorical vs Numerical":
        if not (Categorical or Object) or not Numerical:
            st.info("Need both categorical and numerical columns for this analysis.")
        else:
            cat_cols = Categorical + Object
            col1, col2 = st.columns(2)
            with col1:
                cat_var = st.selectbox("Select Categorical Variable", cat_cols, key="bivar_cat")
            with col2:
                num_var = st.selectbox("Select Numerical Variable", Numerical, key="bivar_num2")
            
            # Box plot by category
            st.markdown("**Box Plot by Category**")
            
            # Limit categories for better visualization
            top_categories = df[cat_var].value_counts().head(10).index
            df_filtered = df[df[cat_var].isin(top_categories)]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            df_filtered.boxplot(column=num_var, by=cat_var, ax=ax)
            ax.set_xlabel(cat_var)
            ax.set_ylabel(num_var)
            ax.set_title(f'{num_var} Distribution by {cat_var}')
            plt.suptitle('')  # Remove automatic title
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            # Group statistics
            st.markdown("**Statistics by Category**")
            grouped_stats = df.groupby(cat_var)[num_var].agg(['mean', 'median', 'std', 'count']).round(2)
            grouped_stats = grouped_stats.sort_values('mean', ascending=False).head(10)
            st.dataframe(grouped_stats, use_container_width=True)
    
    else:  # Categorical vs Categorical
        if len(Categorical + Object) < 2:
            st.info("Need at least 2 categorical columns for this analysis.")
        else:
            cat_cols = Categorical + Object
            col1, col2 = st.columns(2)
            with col1:
                cat_var1 = st.selectbox("Select First Categorical Variable", cat_cols, key="bivar_cat1")
            with col2:
                cat_var2 = st.selectbox("Select Second Categorical Variable", 
                                       [col for col in cat_cols if col != cat_var1], key="bivar_cat2")
            
            # Cross-tabulation
            st.markdown("**Cross-Tabulation Heatmap**")
            
            # Limit to top categories
            top_cat1 = df[cat_var1].value_counts().head(8).index
            top_cat2 = df[cat_var2].value_counts().head(8).index
            df_filtered = df[df[cat_var1].isin(top_cat1) & df[cat_var2].isin(top_cat2)]
            
            crosstab = pd.crosstab(df_filtered[cat_var1], df_filtered[cat_var2])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
            ax.set_title(f'{cat_var1} vs {cat_var2}')
            st.pyplot(fig)
            
            st.markdown("**Frequency Table**")
            st.dataframe(crosstab, use_container_width=True)
    
    # ============================================
    # SECTION 4: CORRELATION MATRIX
    # ============================================
    if Numerical and len(Numerical) > 1:
        st.markdown("---")
        st.markdown("### 🌡️ Correlation Matrix")
        st.caption("Discover linear relationships between all numerical variables")
        st.markdown("")
        
        corr = df[Numerical].corr(method='pearson')
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap, center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix (Pearson)', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        
        # Highlight strong correlations
        st.markdown("**🔍 Strong Correlations (|r| > 0.7):**")
        st.caption("Variables with high linear correlation (positive or negative)")
        strong_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.7:
                    strong_corr.append({
                        'Variable 1': corr.columns[i],
                        'Variable 2': corr.columns[j],
                        'Correlation': f"{corr.iloc[i, j]:.3f}"
                    })
        
        if strong_corr:
            st.dataframe(pd.DataFrame(strong_corr), hide_index=True, use_container_width=True)
        else:
            st.info("No strong correlations found (threshold: |r| > 0.7)")
    
    # ============================================
    # SECTION 5: FILTERED ANALYSIS
    # ============================================
    st.markdown("---")
    st.markdown("### 🔎 Filtered Deep Dive")
    st.caption("Filter data by category and compare against the overall dataset")
    st.markdown("")
    
    if not (Categorical or Object):
        st.info("No categorical columns available for filtering.")
    elif not Numerical:
        st.info("No numerical columns available for analysis.")
    else:
        cat_cols = Categorical + Object
        
        col1, col2 = st.columns(2)
        with col1:
            filter_cat = st.selectbox("Select Category to Filter By", cat_cols, key="filter_cat")
        with col2:
            category_values = df[filter_cat].unique()
            selected_value = st.selectbox(f"Select {filter_cat}", category_values, key="filter_val")
        
        # Filter data
        filtered_df = df[df[filter_cat] == selected_value]
        
        st.info(f"📊 Showing analysis for **{selected_value}** ({len(filtered_df)} rows)")
        
        # Statistics
        st.markdown("**Statistical Summary (Filtered Data)**")
        st.dataframe(filtered_df[Numerical].describe(), use_container_width=True)
        
        # Select numerical column to visualize
        num_col_filtered = st.selectbox("Select Numerical Column to Visualize", Numerical, key="filter_num")
        
        # Comparison visualization
        st.markdown(f"**{num_col_filtered} Distribution: Filtered vs Overall**")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Filtered data
        ax1.hist(filtered_df[num_col_filtered].dropna(), bins=20, edgecolor='black', 
                alpha=0.7, color='steelblue')
        ax1.set_xlabel(num_col_filtered)
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Filtered: {selected_value}')
        ax1.grid(True, alpha=0.3)
        
        # Overall data
        ax2.hist(df[num_col_filtered].dropna(), bins=20, edgecolor='black', 
                alpha=0.7, color='coral')
        ax2.set_xlabel(num_col_filtered)
        ax2.set_ylabel('Frequency')
        ax2.set_title('Overall Dataset')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Comparison metrics
        col_a, col_b, col_c = st.columns(3)
        filtered_mean = filtered_df[num_col_filtered].mean()
        overall_mean = df[num_col_filtered].mean()
        diff = ((filtered_mean - overall_mean) / overall_mean * 100) if overall_mean != 0 else 0
        
        col_a.metric("Filtered Mean", f"{filtered_mean:.2f}", 
                    delta=f"{diff:+.1f}% vs overall")
        col_b.metric("Overall Mean", f"{overall_mean:.2f}")
        col_c.metric("Sample Size", len(filtered_df))
