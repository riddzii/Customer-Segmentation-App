import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from fpdf import FPDF
from streamlit_folium import st_folium

# --- CSS Styling ---
st.markdown("""
<style>
    body {
        background-color: #f5f5dc; /* Beige background */
        font-family: 'Algerian', sans-serif;
    }
    h1 {
        color: #4A4A4A;
        font-size: 36px;
        text-align: center;
        text-decoration: underline;
    }
    h2 {
        color: #333;
        font-size: 28px;
        text-decoration: underline;
    }
    h3 {
        color: #666;
        font-size: 24px;
        font-weight: bold;
    }
    .streamlit-expander {
        background-color: #e7e7e7;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #ff7f50;
        color: white;
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.9);
    }
</style>
""", unsafe_allow_html=True)

# --- Title and Description ---
st.title("🏬 Customer Segmentation App")
st.write("This app segments customers based on Age, Annual Income, Spending Score, and Gender.")

# --- User Authentication ---
def login():
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if 'users' in st.session_state and username in st.session_state['users']:
            if st.session_state['users'][username] == password:
                st.session_state['logged_in'] = True
                st.success("Login successful!")
            else:
                st.error("Incorrect password.")
        else:
            st.error("Invalid username.")

def signup():
    st.subheader("📝 Sign Up")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    
    if st.button("Sign Up"):
        if 'users' not in st.session_state:
            st.session_state['users'] = {}
        
        if new_username in st.session_state['users']:
            st.warning("Username already exists. Please choose a different one.")
        else:
            st.session_state['users'][new_username] = new_password
            st.success("Signup successful! You can now log in.")

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login()
    st.markdown("Or")
    signup()
else:
    st.success("You're logged in! Now you can use the app features.")
    
    # --- Main Application after login ---
    st.subheader("📷 Step 1: Convert Graph Image to Dataset")

    # Function to process the image and extract data
    def extract_data_from_graph(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        data_points = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            data_points.append([x, y])
        df = pd.DataFrame(data_points, columns=['X', 'Y'])
        return df

    # Upload image
    uploaded_image = st.file_uploader("Upload an image of the graph", type=["jpg", "png"])

    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))
        st.image(image, caption="Uploaded Graph Image", use_column_width=True)
        extracted_data = extract_data_from_graph(image)
        st.write("Extracted Dataset:")
        st.write(extracted_data)

    # --- Step 2: Data Upload ---
    st.subheader("🔄 Step 2: Upload Your Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            customer_data = pd.read_csv(uploaded_file)
            st.write("Customer Data", customer_data)
            st.write("DataFrame Shape:", customer_data.shape)
            st.write("DataFrame Columns:", customer_data.columns.tolist())
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    else:
        np.random.seed(42)
        customer_data = pd.DataFrame({
            'Age': np.random.randint(18, 65, size=100),
            'Annual_Income': np.random.randint(20000, 100000, size=100),
            'Spending_Score': np.random.randint(1, 100, size=100),
            'Gender': np.random.choice(['Male', 'Female'], size=100)
        })
        st.write("Generated Customer Data", customer_data)

    # --- Step 3: Feature Selection ---
    st.subheader("🔍 Step 3: Feature Selection")
    features = st.multiselect("Select Features for Clustering", options=customer_data.columns.tolist())

    # --- Step 4: KMeans Clustering ---
    st.subheader("🤖 Step 4: KMeans Clustering")
    n_clusters = st.slider("Choose number of clusters", min_value=2, max_value=10, value=3)

    if features:
        # Prepare features for clustering
        categorical_features = ['Gender'] if 'Gender' in features else []
        numeric_features = [f for f in features if f not in categorical_features]
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )

        # Fit the KMeans model
        try:
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('kmeans', KMeans(n_clusters=n_clusters, random_state=42))])
            customer_data['Segment'] = pipeline.fit_predict(customer_data[features])
            st.write("Segmented Data", customer_data)

            if 'Segment' in customer_data.columns:
                st.write("Segment Centers (first few rows):", pipeline.named_steps['kmeans'].cluster_centers_)

                # --- Step 5: Data Visualization ---
                st.subheader("📊 Step 5: Data Visualization")

                # Scatter Plot for Feature Relationships
                st.write("### Scatter Plot of Features")
                if len(numeric_features) >= 2:
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=customer_data, x=numeric_features[0], y=numeric_features[1], hue='Segment', palette='viridis', ax=ax)
                    ax.set_title(f'Scatter Plot between {numeric_features[0]} and {numeric_features[1]}')
                    st.pyplot(fig)

                # Correlation Matrix Heatmap
                st.write("### Correlation Matrix Heatmap")
                correlation = customer_data[numeric_features + ['Segment']].corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True)
                plt.title("Correlation Matrix")
                st.pyplot(plt)

                # Box Plot for Spending Score by Segment
                if 'Spending_Score' in numeric_features:
                    st.write("### Box Plot of Spending Score by Segment")
                    fig, ax = plt.subplots()
                    sns.boxplot(x='Segment', y='Spending_Score', data=customer_data, palette='viridis', ax=ax)
                    ax.set_title('Spending Score Distribution by Segment')
                    st.pyplot(fig)

                # 3D Scatter Plot
                if len(numeric_features) == 3:
                    st.write("### 3D Scatter Plot of Clusters")
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(customer_data[numeric_features[0]], customer_data[numeric_features[1]], customer_data[numeric_features[2]], 
                               c=customer_data['Segment'], cmap='viridis')
                    ax.set_xlabel(numeric_features[0])
                    ax.set_ylabel(numeric_features[1])
                    ax.set_zlabel(numeric_features[2])
                    st.pyplot(fig)

                # Folium Map (dummy data for demonstration)
                st.subheader("🗺 Customer Locations on Map")
                if 'Age' in customer_data.columns and 'Annual_Income' in customer_data.columns:
                    customer_map = folium.Map(location=[customer_data['Age'].mean(), customer_data['Annual_Income'].mean()], zoom_start=5)
                    MarkerCluster().add_to(customer_map)

                    for i in range(len(customer_data)):
                        folium.Marker(
                            location=[customer_data['Age'].iloc[i], customer_data['Annual_Income'].iloc[i]],
                            popup=f"Segment: {customer_data['Segment'].iloc[i]}"
                        ).add_to(customer_map)

                    st.write("### Customer Locations")
                    st_folium(customer_map, width=700, height=500)
                else:
                    st.warning("Age or Annual Income data is not available for mapping.")

                # --- Step 6: Outlier Detection ---
                st.subheader("🔍 Step 6: Outlier Detection")
                isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = isolation_forest.fit_predict(preprocessor.fit_transform(customer_data[features]))
                customer_data['Outlier'] = outliers
                st.write("Outlier Detection Results", customer_data)

                # --- Step 7: Export Results as PDF ---
                st.subheader("📄 Export Results as PDF")
                if st.button("Download PDF Report"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="Customer Segmentation Report", ln=True, align='C')
                    pdf.cell(200, 10, txt="Segmented Customer Data:", ln=True)
                    pdf.multi_cell(0, 10, txt=customer_data.to_string())
                    pdf_file = "customer_segmentation_report.pdf"
                    pdf.output(pdf_file)
                    with open(pdf_file, "rb") as f:
                        st.download_button("Download PDF", f, file_name=pdf_file)

        except ValueError as e:
            st.error(f"Error in KMeans clustering: {e}")
    else:
        st.warning("Please select features for clustering.")