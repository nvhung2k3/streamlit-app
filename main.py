import streamlit as st
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    st.subheader("Data Exploring")
    st.write("Number of Rows:", str(df.shape[0]))
    st.write("Number of Columns:", str(df.shape[1]))
    st.write("Data Types:")
    st.write(df.dtypes)
    st.write("Data Head:")
    num_rows = st.slider("Number of Rows to Display", 1, len(df), 5)
    st.write(df.head(num_rows))


#visualization
def plot_correlation(df):
    st.write("Plot Correlation between Numeric and Object Columns:")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    object_columns = df.select_dtypes(include=['object']).columns

   # Plot correlation between numeric column and object column
    for num_col in numeric_columns:
        for obj_col in object_columns:
            if len(df[obj_col].unique()) <= 10:
                fig, ax = plt.subplots()
                if df[obj_col].nunique() > 2:
                    sns.boxplot(x=obj_col, y=num_col, data=df, ax=ax)
                else:
                    sns.barplot(x=obj_col, y=num_col, data=df, ax=ax)
                ax.set_title(f"{num_col} vs {obj_col}")
                st.pyplot(fig)

    # Plot correlation between object column
    for obj_col in object_columns:
        if len(df[obj_col].unique()) <= 10:
            if df[obj_col].nunique() >= 2:
                fig, ax = plt.subplots()
                counts = df[obj_col].value_counts()
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
                ax.set_title(f"Distribution of {obj_col}")
                st.pyplot(fig)

    # Plot correlation between numeric columns
    for i in range(len(numeric_columns)):
        for j in range(i+1, len(numeric_columns)):
            fig, ax = plt.subplots()
            sns.scatterplot(x=numeric_columns[i], y=numeric_columns[j], data=df, ax=ax)
            ax.set_title(f"{numeric_columns[i]} vs {numeric_columns[j]}")
            st.pyplot(fig)



# function for check and remove missing value
def handle_missing_values(df):
    st.write("Replace Null Values with Mean or Median:")
    columns_to_drop = []
    for column in df.columns:
        if column in df.columns and df[column].isnull().sum() > 0:
            #numric column
            if df[column].dtype in ['int64', 'float64']:
                mean_value = df[column].mean()
                median_value = df[column].median()
                #select mean or median
                st.write(f"Column: {column}")
                st.write(f"Mean Value: {mean_value}, Median Value: {median_value}")
                fill_with = st.radio(f"Fill {column} with Mean or Median:", ["Mean", "Median"], key=column)
                if fill_with == "Mean":
                    df[column].fillna(mean_value, inplace=True)
                elif fill_with == "Median":
                    df[column].fillna(median_value, inplace=True)
            else:
                unique_values = df[column].unique()
                if len(unique_values) <= 10:
                    
                    mapping = {val: i+1 for i, val in enumerate(unique_values)}
                    df[column] = df[column].map(mapping)

                    mean_value = df[column].mean()
                    median_value = df[column].median()

                    st.write(f"Column: {column}")
                    st.write(f"Mean Value: {mean_value}, Median Value: {median_value}")
                    fill_with = st.radio(f"Fill {column} with Mean or Median:", ["Mean", "Median"], key=column)
                    if fill_with == "Mean":
                        df[column].fillna(mean_value, inplace=True)
                    elif fill_with == "Median":
                        df[column].fillna(median_value, inplace=True)
                else:
                    #if not numeric and object have unique value more than 10
                    columns_to_drop.append(column)

    # Drop columns that do not meet the conditions
    df.drop(columns_to_drop, axis=1, inplace=True)

    st.write("Updated DataFrame:")
    st.write(df)


#check and remove outlier
def handle_outliers(df):
    st.write("Check and Handle Outliers:")
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            st.write(f"Column: {column}")
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            st.write(f"Number of Outliers: {len(outliers)}")
            st.write(f"Outliers:")
            st.write(outliers)
            # Replace outliers with the upper/lower bound
            df.loc[df[column] < lower_bound, column] = lower_bound
            df.loc[df[column] > upper_bound, column] = upper_bound

    st.write("Updated DataFrame after handling outliers:")
    st.write(df)


#convert object to numeric
def encode_object_columns(df):
    st.write("Encode Object Columns with Few Unique Values:")
    columns_to_encode = []
    for column in df.columns:

        if df[column].dtype == 'object' and len(df[column].unique()) <= 10:
            columns_to_encode.append(column)


    if len(columns_to_encode) == 0:
        st.write("No object columns with few unique values found.")
        return df

    encoder = ce.OrdinalEncoder()
    df[columns_to_encode] = encoder.fit_transform(df[columns_to_encode])

    st.write("Encoded DataFrame:")
    st.write(df)

    return df



# Matruc to correclation
def plot_correlation_matrix(df):
    encoded_df = encode_object_columns(df)  # Encode object columns
    numeric_columns = encoded_df.select_dtypes(include=['float64', 'int64']).columns

    st.write("Plot Correlation Matrix:")
    corr_matrix = encoded_df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    

def train_model(df, selected_model):
    st.sidebar.title("Select X and Y Columns")
    x_columns = st.sidebar.multiselect("Select X Columns", df.columns)
    y_column = st.sidebar.selectbox("Select Y Column", df.columns)

    st.write("Selected X Columns:")
    st.write(x_columns)
    st.write("Selected Y Column:")
    st.write(y_column)

    if st.sidebar.button("Train Model"):
        df = encode_object_columns(df)
        X = df[x_columns]
        y = df[y_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = model_variables[selected_model]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Values')
        st.pyplot(fig)

# page header
st.set_page_config(page_title='Model Training', page_icon=':books:', layout='wide')

# sidebar for file browser
uploaded_file = st.sidebar.file_uploader("Choose a file")

# sidebar for model selection
st.sidebar.title("Model Selection")
models = ["Linear Regression", "Random Forest"]
selected_model = st.sidebar.radio("Model", models, index=0)

model_variables = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
}

if selected_model in model_variables:
    model = model_variables[selected_model]

st.title('DATA ANALYSIS AND TRAIN MODEL')
# st.markdown('This app allows you to train and evaluate different regression models.')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    explore_data(df)
    handle_missing_values(df)
    handle_outliers(df)
    plot_correlation(df)
    plot_correlation_matrix(df)
    train_model(df, selected_model)
