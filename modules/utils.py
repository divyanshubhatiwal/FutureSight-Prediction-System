import numpy as np 
import pandas as pd 
from pandas.api.types import is_numeric_dtype
# from pandas_profiling import ProfileReport

def isCategorical(col):
    unis = np.unique(col)
    if len(unis)<0.2*len(col):
        return True
    return False

# def getProfile(data):
#     report = ProfileReport(data)
#     report.to_file(output_file = 'data/output.html')

def getColumnTypes(cols):
    Categorical=[]
    Numerical = []
    Object = []
    for i in range(len(cols)):
        if cols["type"][i]=='categorical':
            Categorical.append(cols['column_name'][i])
        elif cols["type"][i]=='numerical':
            Numerical.append(cols['column_name'][i])
        else:
            Object.append(cols['column_name'][i])
    return Categorical, Numerical, Object

def isNumerical(col):
    return is_numeric_dtype(col)

def genMetaData(df):
    col = df.columns
    ColumnType = [] 
    Categorical = []
    Object = []
    Numerical = []
    for i in range(len(col)):
        if isCategorical(df[col[i]]):
            ColumnType.append((col[i],"categorical"))
            Categorical.append(col[i])
        
        elif is_numeric_dtype(df[col[i]]):
            ColumnType.append((col[i],"numerical"))
            Numerical.append(col[i])
        
        else:
            ColumnType.append((col[i],"object"))
            Object.append(col[i])

    return ColumnType

def clean_string_columns(df):
    """
    Clean string columns by stripping whitespace and handling null values.
    Does NOT convert types - preserves categorical strings for visualization.
    Returns: tuple (cleaned_df, cleaning_summary)
    """
    summary = {
        'cleaned_columns': [],
        'whitespace_cleaned': 0,
        'empty_to_null': 0
    }
    
    df_cleaned = df.copy()
    
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':  # String columns
            original = df_cleaned[col].copy()
            
            # Strip whitespace
            df_cleaned[col] = df_cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            
            # Count whitespace cleanings
            whitespace_changes = (original != df_cleaned[col]).sum()
            if whitespace_changes > 0:
                summary['whitespace_cleaned'] += whitespace_changes
            
            # Convert empty strings to NaN
            empty_count = (df_cleaned[col] == '').sum()
            df_cleaned[col] = df_cleaned[col].replace('', np.nan)
            if empty_count > 0:
                summary['empty_to_null'] += empty_count
            
            summary['cleaned_columns'].append(col)
    
    return df_cleaned, summary

def assess_data_quality(df):
    """
    Assess data quality for each column.
    Returns: DataFrame with quality metrics for each column
    """
    quality_report = []
    
    for col in df.columns:
        col_data = df[col]
        total_count = len(col_data)
        null_count = col_data.isnull().sum()
        null_pct = (null_count / total_count) * 100
        unique_count = col_data.nunique()
        unique_pct = (unique_count / total_count) * 100
        
        # Quality indicator
        if null_pct > 50:
            quality = "✗ Poor"
        elif null_pct > 30 or unique_pct > 95:
            quality = "⚠ Warning"
        else:
            quality = "✓ Good"
        
        # Column-specific stats
        col_type = str(col_data.dtype)
        stats = {}
        
        if is_numeric_dtype(col_data):
            stats['mean'] = f"{col_data.mean():.2f}" if null_count < total_count else "N/A"
            stats['std'] = f"{col_data.std():.2f}" if null_count < total_count else "N/A"
            # Outliers (values beyond 3 std devs)
            if null_count < total_count:
                mean_val = col_data.mean()
                std_val = col_data.std()
                outliers = ((col_data < mean_val - 3*std_val) | (col_data > mean_val + 3*std_val)).sum()
                stats['outliers'] = outliers
            else:
                stats['outliers'] = 0
        else:
            # For categorical/object columns
            top_values = col_data.value_counts().head(3)
            stats['top_values'] = ", ".join([f"{val} ({cnt})" for val, cnt in top_values.items()])
        
        quality_report.append({
            'column': col,
            'type': col_type,
            'null_count': null_count,
            'null_pct': null_pct,
            'unique_count': unique_count,
            'unique_pct': unique_pct,
            'quality': quality,
            'stats': stats
        })
    
    return pd.DataFrame(quality_report)

def assess_ml_readiness(df, metadata_df):
    """
    Assess which columns are ready for ML and which need attention.
    Returns: dict with 'ready', 'needs_attention' lists and reasons
    """
    ready_columns = []
    needs_attention = []
    
    for _, row in metadata_df.iterrows():
        col_name = row['column_name']
        col_type = row['type']
        
        if col_name not in df.columns:
            continue
        
        col_data = df[col_name]
        null_pct = (col_data.isnull().sum() / len(col_data)) * 100
        unique_pct = (col_data.nunique() / len(col_data)) * 100
        
        issues = []
        
        # Check for issues
        if null_pct > 50:
            issues.append(f"High nulls ({null_pct:.1f}%)")
        if unique_pct > 95 and col_type != 'object':
            issues.append(f"Too many unique values ({unique_pct:.1f}%)")
        if col_data.nunique() == len(col_data):
            issues.append("All unique (likely ID column)")
        if is_numeric_dtype(col_data) and col_data.std() == 0:
            issues.append("Zero variance")
        
        if issues:
            needs_attention.append({
                'column': col_name,
                'type': col_type,
                'issues': ", ".join(issues)
            })
        else:
            ready_columns.append({
                'column': col_name,
                'type': col_type,
                'null_pct': null_pct
            })
    
    return {
        'ready': ready_columns,
        'needs_attention': needs_attention
    }

def identify_useless_columns(df, threshold_nulls=0.8, threshold_unique=0.95):
    """
    Identify columns that are likely useless for ML:
    - High percentage of null values
    - All unique values for non-numeric columns (like IDs, names)
    - Sequential ID columns (1, 2, 3, ... n)
    - Zero variance (constant values)
    Returns: dict with useless columns and reasons
    """
    useless = []
    
    for col in df.columns:
        col_data = df[col]
        total_count = len(col_data)
        null_pct = col_data.isnull().sum() / total_count
        unique_pct = col_data.nunique() / total_count
        
        reasons = []
        
        # High nulls
        if null_pct > threshold_nulls:
            reasons.append(f"{null_pct*100:.1f}% null values")
        
        # Check for ID-like patterns
        if unique_pct >= 1.0:
            # All values are unique
            if is_numeric_dtype(col_data):
                # Check if it's a sequential ID (1, 2, 3, ... n)
                non_null = col_data.dropna().astype(int) if col_data.dtype in ['int64', 'float64'] else col_data.dropna()
                if len(non_null) > 0 and list(non_null) == list(range(1, len(non_null) + 1)):
                    reasons.append("Sequential ID column (1, 2, 3, ...)")
                # Otherwise, numeric columns with all unique values are OKAY (e.g., prices, scores)
            else:
                # For text columns, all unique likely means names/IDs
                reasons.append("All values unique (likely ID or name column)")
        elif unique_pct > threshold_unique:
            # Very high uniqueness (>95%)
            if not is_numeric_dtype(col_data):
                # Only flag non-numeric columns with very high uniqueness
                reasons.append(f"{unique_pct*100:.1f}% unique values (likely ID/name)")
        
        # Zero variance for numeric columns
        if is_numeric_dtype(col_data) and col_data.std() == 0:
            reasons.append("Zero variance (constant value)")
        
        if reasons:
            useless.append({
                'column': col,
                'reasons': reasons
            })
    
    return useless

def makeMapDict(col): 
    uniqueVals = list(np.unique(col))
    uniqueVals.sort()
    dict_ = {uniqueVals[i]: i for i in range(len(uniqueVals))}
    return dict_

def mapunique(df, colName):
    dict_ = makeMapDict(df[colName])
    cat = np.unique(df[colName])
    df[colName] =  df[colName].map(dict_)
    return cat 

if __name__ == '__main__':
    df = {"Name": ["salil", "saxena", "for", "int"]}
    df = pd.DataFrame(df)
    print("Mapping dict: ", makeMapDict(df["Name"]))
    print("original df: ")
    print(df.head())
    pp = mapunique(df, "Name")
    print("New df: ")
    print(pp.head())