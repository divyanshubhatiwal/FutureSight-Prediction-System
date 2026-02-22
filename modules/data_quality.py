# Load important libraries 
import pandas as pd
import streamlit as st 
import os
from modules import utils

def app():
    """Enhanced data quality and metadata management page.
    Shows data quality overview, ML readiness assessment, and allows column type changes.
    """

    # Load the uploaded data 
    if 'main_data.csv' not in os.listdir('data'):
        st.warning("⚠️ **No data found!**")
        st.info("📂 Please upload your dataset through the **Upload Data** page first.")
    else:
        # Mark this page as completed when accessed
        if 'completed_pages' not in st.session_state:
            st.session_state.completed_pages = set()
        st.session_state.completed_pages.add('Data Quality & Metadata')
        
        data = pd.read_csv('data/main_data.csv')
        col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')
        
        st.markdown("## 📊 Data Quality & Metadata")
        st.caption("Review data quality, assess ML readiness, and modify column types")
        st.markdown("")
        
        # ============================================
        # SECTION 1: DATA QUALITY OVERVIEW
        # ============================================
        st.markdown("---")
        st.markdown("### 📊 Data Quality Overview")
        
        with st.spinner("Analyzing data quality..."):
            quality_df = utils.assess_data_quality(data)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        good_count = (quality_df['quality'] == "✓ Good").sum()
        warning_count = (quality_df['quality'] == "⚠ Warning").sum()
        poor_count = (quality_df['quality'] == "✗ Poor").sum()
        
        col1.metric("📊 Total Columns", len(quality_df))
        col2.metric("✅ Good Quality", good_count)
        col3.metric("⚠️ Warnings", warning_count)
        col4.metric("❌ Poor Quality", poor_count)
        
        # Overall quality indicator
        quality_score = (good_count / len(quality_df) * 100) if len(quality_df) > 0 else 0
        st.markdown("")
        
        if quality_score >= 80:
            st.success(f"🎉 **Excellent Data Quality:** {quality_score:.0f}% of columns are in good condition!")
        elif quality_score >= 60:
            st.info(f"📊 **Good Data Quality:** {quality_score:.0f}% of columns are in good condition.")
        elif quality_score >= 40:
            st.warning(f"⚠️ **Fair Data Quality:** {quality_score:.0f}% of columns need attention.")
        else:
            st.error(f"❌ **Poor Data Quality:** Only {quality_score:.0f}% of columns are in good condition.")
        
        st.markdown("")
        # Detailed quality table
        st.markdown("#### 🔍 Column-by-Column Analysis")
        
        # Display quality info for each column
        for idx, row in quality_df.iterrows():
            with st.expander(f"{row['quality']} **{row['column']}** ({row['type']})"):
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.write("**Basic Info:**")
                    st.write(f"- Null values: {row['null_count']} ({row['null_pct']:.1f}%)")
                    st.write(f"- Unique values: {row['unique_count']} ({row['unique_pct']:.1f}%)")
                
                with col_right:
                    st.write("**Statistics:**")
                    stats = row['stats']
                    if 'mean' in stats:
                        st.write(f"- Mean: {stats['mean']}")
                        st.write(f"- Std Dev: {stats['std']}")
                        st.write(f"- Outliers: {stats['outliers']}")
                    elif 'top_values' in stats:
                        st.write(f"- Top values: {stats['top_values']}")
        
        # ============================================
        # SECTION 2: ML READINESS ASSESSMENT
        # ============================================
        st.markdown("---")
        st.markdown("### 🤖 ML Readiness Assessment")
        
        ml_assessment = utils.assess_ml_readiness(data, col_metadata)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("#### ✅ Ready for ML")
            st.write(f"**{len(ml_assessment['ready'])} columns** are ready for machine learning:")
            
            if ml_assessment['ready']:
                ready_df = pd.DataFrame(ml_assessment['ready'])
                st.dataframe(
                    ready_df[['column', 'type', 'null_pct']].style.format({'null_pct': '{:.1f}%'}),
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("No columns are ready for ML. Please review data quality.")
        
        with col_b:
            st.markdown("#### ⚠️ Needs Attention")
            st.write(f"**{len(ml_assessment['needs_attention'])} columns** need attention:")
            
            if ml_assessment['needs_attention']:
                attention_df = pd.DataFrame(ml_assessment['needs_attention'])
                st.dataframe(
                    attention_df[['column', 'type', 'issues']],
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.success("All columns are ready!")
        
        # ============================================
        # SECTION 3: CHANGE COLUMN TYPES (Collapsible)
        # ============================================
        st.markdown("---")
        with st.expander("⚙️ **Advanced: Change Column Types**", expanded=False):
            st.markdown("### Change Column Metadata")
            st.write("Modify column types if the automatic detection was incorrect.")
            
            # Tab selection for single or batch mode
            tab1, tab2 = st.tabs(["Single Column", "Batch Mode"])
            
            # ---- Single Column Mode ----
            with tab1:
                col1, col2 = st.columns(2)
                
                # Design column 1 
                name = col1.selectbox("Select Column", data.columns, key="single_col")
                
                # Design column 2
                current_type = col_metadata[col_metadata['column_name'] == name]['type'].values[0]
                column_options = ['numerical', 'categorical', 'object']
                current_index = column_options.index(current_type)
                
                new_type = col2.selectbox("Select Column Type", options=column_options, index=current_index, key="single_type")
                
                st.write(f"Current type: **{current_type}** → New type: **{new_type}**")
                
                if st.button("Apply Change", key="single_apply"):
                    col_metadata.loc[col_metadata['column_name'] == name, 'type'] = new_type
                    col_metadata.to_csv('data/metadata/column_type_desc.csv', index=False)
                    st.success(f"✓ Changed **{name}** to **{new_type}**")
                    st.rerun()
            
            # ---- Batch Mode ----
            with tab2:
                st.write("Select multiple columns and apply the same type to all of them.")
                
                selected_columns = st.multiselect(
                    "Select Columns",
                    options=data.columns.tolist(),
                    key="batch_cols"
                )
                
                if selected_columns:
                    batch_type = st.selectbox(
                        "Select Type to Apply",
                        options=['numerical', 'categorical', 'object'],
                        key="batch_type"
                    )
                    
                    st.write(f"Will change **{len(selected_columns)} columns** to **{batch_type}**")
                    
                    if st.button("Apply to All Selected", key="batch_apply"):
                        for col in selected_columns:
                            col_metadata.loc[col_metadata['column_name'] == col, 'type'] = batch_type
                        col_metadata.to_csv('data/metadata/column_type_desc.csv', index=False)
                        st.success(f"✓ Changed {len(selected_columns)} columns to **{batch_type}**")
                        st.rerun()
        
        # Display current metadata at bottom
        st.markdown("---")
        st.markdown("### 📋 Current Metadata")
        st.dataframe(col_metadata, hide_index=True, use_container_width=True)
