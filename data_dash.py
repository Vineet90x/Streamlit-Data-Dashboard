import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Data Dashboard")

# Display the uploaded file
uploaded_file = st.file_uploader("Select your files", type='csv')

def download_link(df, filename="cleaned_data.csv", text="Download cleaned data"):
    csv = df.to_csv(index=False)
    b64 = st.download_button(label=text, 
                             data=csv,
                             file_name=filename,
                             mime='text/csv')
    return b64


if uploaded_file is not None:
    
    try:
        # Check if the file is empty
        if uploaded_file.read(1):
            uploaded_file.seek(0)  # Reset file pointer to the beginning
            try:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')  # or another encoding if needed
            except pd.errors.EmptyDataError:
                st.error("The uploaded file is empty. Please upload a valid CSV file.")
                df = pd.DataFrame()  # Empty DataFrame for error handling
            except pd.errors.ParserError:
                st.error("Error parsing the file. Please upload a valid CSV file.")
                df = pd.DataFrame()  # Empty DataFrame for error handling
        else:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
            df = pd.DataFrame()  # Empty DataFrame for error handling

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        df = pd.DataFrame()  # Empty DataFrame for error handling




    st.subheader("Data preview")
    st.write(df)
    num_rows, num_cols = df.shape
    st.write(f"Rows: {num_rows}  x  Columns: {num_cols}")


    st.subheader("Data Cleaning")
    
    # Handling missing data
    st.markdown("### Handle Missing Data")
    missing_data_action = st.selectbox("Select an action for missing data", 
                                       ["Do nothing", "Drop missing rows", "Fill missing with a value", 
                                        "Forward fill", "Backward fill"])
    
    if missing_data_action == "Drop missing rows":
        df = df.dropna()
        st.success("Missing data rows dropped.")
    elif missing_data_action == "Fill missing with a value":
        fill_value = st.text_input("Enter value to fill missing data with:")
        if fill_value:
            df = df.fillna(fill_value)
            st.success(f"Missing data filled with {fill_value}.")
    elif missing_data_action == "Forward fill":
        df = df.fillna(method='ffill')
        st.success("Missing data forward-filled.")
    elif missing_data_action == "Backward fill":
        df = df.fillna(method='bfill')
        st.success("Missing data backward-filled.")

    # Option to view the cleaned data
    st.subheader("Cleaned Data Preview")
    st.write(df)
    num_rows, num_cols = df.shape
    st.write(f"Rows: {num_rows}  x  Columns: {num_cols}")
    download_link(df)

    st.subheader("Data summary")
    st.write(df.describe())

    st.subheader("Filter Data")
    column = df.columns.tolist()
    select_column = st.selectbox("Select column to filter by", column)
    unique_values = df[select_column].unique()
    select_value = st.selectbox("Select value to filter by", unique_values)

    filtered_df = df[df[select_column] == select_value]
    st.write(filtered_df)

    
    st.subheader("Advanced Filtering")

    # Multi-column filtering
    st.markdown("### Filter by Multiple Columns")

    # Select columns to filter
    filter_columns = st.multiselect("Select columns to filter by", df.columns.tolist())
    filters = {}

    # Collect filter criteria for each selected column
    for column in filter_columns:
        if df[column].dtype == 'object':  # For categorical columns
            unique_values = df[column].unique()
            selected_values = st.multiselect(f"Select values for {column}", unique_values)
            if selected_values:
                filters[column] = selected_values
        elif pd.api.types.is_numeric_dtype(df[column]):  # For numerical columns
            min_value, max_value = st.slider(f"Select range for {column}", float(df[column].min()), float(df[column].max()), (float(df[column].min()), float(df[column].max())))
            filters[column] = (min_value, max_value)
    
    # Apply filters
    filtered_df = df.copy()
    for column, criteria in filters.items():
        if isinstance(criteria, list):  # Categorical filter
            filtered_df = filtered_df[filtered_df[column].isin(criteria)]
        elif isinstance(criteria, tuple):  # Numerical range filter
            filtered_df = filtered_df[(filtered_df[column] >= criteria[0]) & (filtered_df[column] <= criteria[1])]
    
    st.write(filtered_df)

    st.subheader("Correlation Matrix and Heatmap")
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    if not numerical_df.empty:
            corr_matrix = numerical_df.corr()
            st.write("Correlation Matrix:")
            st.write(corr_matrix)

            st.write("Correlation Heatmap:")
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(plt)
    else:
            st.write("No numerical data available to compute correlation matrix.")


    st.subheader("Plot Data")
    x_column = st.selectbox("Select x-axis column", column)
    y_column = st.selectbox("Select y-axis column", column)
    plot_type = st.selectbox("Select plot type", ["line", "bar", "scatter"])

    # Proceed with plotting
    if st.button("Generate Plot"):
        if filtered_df.empty:
            st.warning("No data available for the selected filter.")
        else:
            # Convert data types to ensure compatibility for plotting
            df[x_column] = df[x_column].astype(str)  # Convert x_column to string
            df[y_column] = pd.to_numeric(df[y_column], errors='coerce')  # Convert y_column to numeric
        
            # Drop rows with NaN values that could result from conversion issues
            df = df.dropna(subset=[x_column, y_column])
            plt.figure(figsize=(10, 6))

            if plot_type == "line":
                plt.plot(df[x_column], df[y_column], marker='o')
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f"{plot_type.capitalize()} Plot of {y_column} vs {x_column}")
                st.pyplot(plt)
            elif plot_type == "bar":
                plt.bar(df[x_column], df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f"{plot_type.capitalize()} Plot of {y_column} vs {x_column}")
                st.pyplot(plt)
            elif plot_type == "scatter":
                plt.scatter(df[x_column], df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f"{plot_type.capitalize()} Plot of {y_column} vs {x_column}")
                st.pyplot(plt)
            else:
                st.error("Unsupported plot type")
else:
    st.write("Waiting on file upload..")