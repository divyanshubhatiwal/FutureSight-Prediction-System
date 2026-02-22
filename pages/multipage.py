"""
This file is the framework for generating multiple Streamlit applications 
through an object oriented framework. 
"""

# Import necessary libraries 
import streamlit as st
import pandas as pd
import os

# Define the multipage class to manage the multiple apps in our program 
class MultiPage: 
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []
        
        # Initialize session state for completed pages
        if 'completed_pages' not in st.session_state:
            st.session_state.completed_pages = set()
    
    def check_file_based_completion(self):
        """Check file system and auto-mark pages as complete based on file existence"""
        # Check if data files exist and mark pages accordingly
        try:
            if os.path.exists('data') and 'main_data.csv' in os.listdir('data'):
                st.session_state.completed_pages.add('Upload Data')
                # If metadata also exists, mark data quality as viewable
                if os.path.exists('data/metadata/column_type_desc.csv'):
                    # Note: We don't auto-complete Data Quality as user should view it explicitly
                    pass
        except Exception:
            pass
    
    def add_page(self, title, func, requires=None) -> None: 
        """Class Method to Add pages to the project

        Args:
            title ([str]): The title of page which we are adding to the list of apps 
            func: Python function to render this page in Streamlit
            requires ([list], optional): List of page titles that must be completed before this page is accessible
        """
        if requires is None:
            requires = []
            
        self.pages.append(
            {
                "title": title, 
                "function": func,
                "requires": requires
            }
        )
    
    def mark_page_complete(self, title):
        """Mark a page as completed"""
        if 'completed_pages' not in st.session_state:
            st.session_state.completed_pages = set()
        st.session_state.completed_pages.add(title)
    
    def is_page_accessible(self, page):
        """Check if a page's prerequisites are met"""
        requires = page.get('requires', [])
        if not requires:
            return True
        
        # Check if all required pages are completed
        return all(req in st.session_state.completed_pages for req in requires)

    def run(self):
        # Check file system for existing data and mark pages accordingly
        self.check_file_based_completion()
        
        # Display progress indicator
        total_pages = len(self.pages)
        completed_count = len(st.session_state.completed_pages)
        
        st.sidebar.markdown("### 📊 Progress Tracker")
        st.sidebar.progress(completed_count / total_pages if total_pages > 0 else 0)
        st.sidebar.caption(f"{completed_count} of {total_pages} pages completed")
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧭 Navigation")
        st.sidebar.caption("Select a page to get started")
        
        # Create list of page options with status indicators
        page_options = []
        page_labels = []
        
        for page in self.pages:
            title = page.get('title', "")
            is_accessible = self.is_page_accessible(page)
            is_completed = title in st.session_state.completed_pages
            
            # Add status indicator
            if is_completed:
                label = f"✓ {title}"
            elif not is_accessible:
                label = f"🔒 {title}"
            else:
                label = f"  {title}"
            
            page_options.append(page)
            page_labels.append(label)
        
        # Radio button list to select the page to run
        selected_index = st.sidebar.radio(
            'App Navigation', 
            range(len(page_options)),
            format_func=lambda i: page_labels[i]
        )
        
        selected_page = page_options[selected_index]
        
        # Check if page is accessible
        if not self.is_page_accessible(selected_page):
            st.error("🔒 **Page Locked**")
            st.warning("This page requires you to complete the following pages first:")
            
            requirements_df = []
            for req in selected_page.get('requires', []):
                is_complete = req in st.session_state.completed_pages
                requirements_df.append({
                    'Status': '✅ Complete' if is_complete else '⏳ Pending',
                    'Required Page': req
                })
            
            st.dataframe(pd.DataFrame(requirements_df), hide_index=True, use_container_width=True)
            st.info("💡 Complete the pending pages to unlock this section.")
        else:
            # run the app function 
            selected_page.get('function')()
