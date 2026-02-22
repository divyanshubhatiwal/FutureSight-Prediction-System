import streamlit as st
import numpy as np
import pandas as pd
import os
from modules import utils

# @st.cache
def app():
    st.markdown("## 📁 Data Upload")
    st.caption("Upload your dataset to begin the analysis journey")
    st.markdown("")
    
    # Check if data already exists
    data_exists = os.path.exists('data/main_data.csv')
    
    if data_exists:
        st.success("📂 **Data loaded!** Your uploaded data is ready for analysis.")
        col_info1, col_info2, col_info3 = st.columns(3)
        
        try:
            existing_data = pd.read_csv('data/main_data.csv')
            col_info1.metric("Rows", f"{len(existing_data):,}")
            col_info2.metric("Columns", len(existing_data.columns))
            col_info3.metric("Size", f"{existing_data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        except:
            pass
        
        st.markdown("")
        
        # Add option to clear existing data
        with st.expander("⚠️ **Want to start over?**"):
            st.warning("Clearing data will reset all your progress and remove trained models.")
            if st.button("🗑️ Clear All Data & Start Fresh", type="primary"):
                try:
                    if os.path.exists('data/main_data.csv'):
                        os.remove('data/main_data.csv')
                    if os.path.exists('data/metadata/column_type_desc.csv'):
                        os.remove('data/metadata/column_type_desc.csv')
                    # Clear session state
                    if 'completed_pages' in st.session_state:
                        st.session_state.completed_pages = set()
                    st.success("✅ Data cleared successfully! You can now upload new data.")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing data: {e}")

    # Upload the dataset and save as csv
    st.markdown("### 📂 Upload a CSV file for analysis")
    st.info("💡 **Tip:** Make sure your CSV has column headers in the first row.")
    st.write("")

    # Code to read a single file 
    uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
    global data
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            data = pd.read_excel(uploaded_file)
    
    
    
    # uploaded_files = st.file_uploader("Upload your CSV file here.", type='csv', accept_multiple_files=False)
    # # Check if file exists 
    # if uploaded_files:
    #     for file in uploaded_files:
    #         file.seek(0)
    #     uploaded_data_read = [pd.read_csv(file) for file in uploaded_files]
    #     raw_data = pd.concat(uploaded_data_read)
    
    # uploaded_files = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=False)
    # print(uploaded_files, type(uploaded_files))
    # if uploaded_files:
    #     for file in uploaded_files:
    #         file.seek(0)
    #     uploaded_data_read = [pd.read_csv(file) for file in uploaded_files]
    #     raw_data = pd.concat(uploaded_data_read)
    
    # read temp data 
    # data = pd.read_csv('data/2015.csv')


    ''' Load the data and save the columns with categories as a dataframe. 
    This section also allows changes in the numerical and categorical columns. '''
    if st.button("Load Data"):
        
        # Preprocess the data (basic cleaning only)
        st.markdown("### Data Preprocessing")
        with st.spinner("Cleaning data..."):
            data_cleaned, cleaning_summary = utils.clean_string_columns(data)
        
        # Display preprocessing summary
        if cleaning_summary['cleaned_columns']:
            st.success("✅ Data preprocessing completed!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Columns Processed", len(cleaning_summary['cleaned_columns']))
            with col2:
                st.metric("Whitespace Trimmed", cleaning_summary['whitespace_cleaned'])
            with col3:
                st.metric("Empty Values Fixed", cleaning_summary['empty_to_null'])
            
            st.markdown("")
            with st.expander("🔍 View preprocessing details"):
                if cleaning_summary['whitespace_cleaned'] > 0:
                    st.write(f"• Removed leading/trailing whitespace from {cleaning_summary['whitespace_cleaned']} values")
                if cleaning_summary['empty_to_null'] > 0:
                    st.write(f"• Converted {cleaning_summary['empty_to_null']} empty strings to null values")
            
            st.info("📊 **Note:** Categorical strings preserved for visualization. Encoding happens automatically during ML training.")
        else:
            st.success("✅ Data is clean! No preprocessing needed.")
        
        st.markdown("")
        
        # Use cleaned data
        data = data_cleaned
        
        # Raw data 
        st.markdown("### 📊 Data Preview")
        st.caption(f"Showing all {len(data):,} rows and {len(data.columns)} columns")
        st.dataframe(data, use_container_width=True, height=400)
        #utils.getProfile(data)
        #st.markdown("<a href='output.html' download target='_blank' > Download profiling report </a>",unsafe_allow_html=True)
        #HtmlFile = open("data/output.html", 'r', encoding='utf-8')
        #source_code = HtmlFile.read() 
        #components.iframe("data/output.html")# Save the data to a new file 
        data.to_csv('data/main_data.csv', index=False)
        
        #Generate a pandas profiling report
        #if st.button("Generate an analysis report"):
        #    utils.getProfile(data)
            #Open HTML file

        # 	pass

        # Collect the categorical and numerical columns 
        
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
        
        # Save the columns as a dataframe or dictionary
        columns = []

        # Iterate through the numerical and categorical columns and save in columns 
        columns = utils.genMetaData(data) 
        
        # Save the columns as a dataframe with categories
        # Here column_name is the name of the field and the type is whether it's numerical or categorical
        columns_df = pd.DataFrame(columns, columns = ['column_name', 'type'])
        columns_df.to_csv('data/metadata/column_type_desc.csv', index = False)

        # Display columns 
        st.markdown("### 📋 Column Metadata Summary")
        st.caption("Automatically detected column types")
        
        # Create a better display format
        col_display_df = columns_df.copy()
        col_display_df.columns = ['Column Name', 'Data Type']
        col_display_df.index = range(1, len(col_display_df) + 1)
        
        st.dataframe(col_display_df, use_container_width=True)
        
        st.markdown("")
        st.info("🔧 **Need to adjust column types?** Visit the **Data Quality & Metadata** page to make changes.")
        
        # Mark this page as completed
        if 'completed_pages' not in st.session_state:
            st.session_state.completed_pages = set()
        st.session_state.completed_pages.add('Upload Data')
        
        st.markdown("---")
        st.success("🎉 **Upload completed successfully!**")
        st.balloons()
        
        col_next1, col_next2 = st.columns(2)
        with col_next1:
            st.markdown("👉 **Next Steps:**")
            st.markdown("- ✅ Check **Data Quality & Metadata**")
            st.markdown("- 📈 Explore **Data Analysis**")
            st.markdown("- 🤖 Train models in **Machine Learning**")
        with col_next2:
            st.markdown("📊 **Quick Stats:**")
            st.markdown(f"- Rows: **{len(data):,}**")
            st.markdown(f"- Columns: **{len(columns_df)}**")
            st.markdown(f"- Memory: **{data.memory_usage(deep=True).sum() / 1024:.1f} KB**")