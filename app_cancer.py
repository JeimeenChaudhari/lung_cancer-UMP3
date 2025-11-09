"""
Lung Cancer Survival Prediction Dashboard
A modern Streamlit application for EDA and survival prediction
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    layout="wide",
    page_title="Lung Cancer Survival Predictor",
    page_icon="🎗️"
)

# =====================================================
# LOAD MODELS AND DATA
# =====================================================
@st.cache_resource
def load_models():
    """Load the preprocessor and model"""
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('lung_cancer_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return preprocessor, model

@st.cache_data
def load_data():
    """Load the dataset"""
    return pd.read_csv('dataset_med.csv')

try:
    preprocessor, model = load_models()
    df = load_data()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# =====================================================
# CUSTOM CSS FOR MODERN UI
# =====================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2913/2913133.png", width=100)
st.sidebar.title("🎗️ Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["📊 Exploratory Data Analysis", "🤖 Prediction Tool"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**About this App**\n\n"
    "This dashboard provides insights into lung cancer survival "
    "and allows healthcare professionals to predict patient outcomes "
    "based on clinical features."
)

# =====================================================
# PAGE 1: EXPLORATORY DATA ANALYSIS
# =====================================================
if page == "📊 Exploratory Data Analysis":
    st.markdown('<p class="main-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Comprehensive analysis of lung cancer patient data</p>',
        unsafe_allow_html=True
    )
    
    # Dataset Overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Survived", df['survived'].sum())
    with col3:
        st.metric("Not Survived", len(df) - df['survived'].sum())
    with col4:
        survival_rate = (df['survived'].sum() / len(df)) * 100
        st.metric("Survival Rate", f"{survival_rate:.1f}%")
    
    st.markdown("---")
    
    # Display statistics
    with st.expander("📈 View Detailed Statistics"):
        st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Visualization Section
    st.subheader("📊 Interactive Visualizations")
    
    # Row 1: Two charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart 1: Survival Rate by Cancer Stage
        if 'cancer_stage' in df.columns:
            survival_by_stage = df.groupby('cancer_stage')['survived'].agg(['sum', 'count'])
            survival_by_stage['survival_rate'] = (survival_by_stage['sum'] / survival_by_stage['count']) * 100
            survival_by_stage = survival_by_stage.reset_index()
            
            fig1 = px.bar(
                survival_by_stage,
                x='cancer_stage',
                y='survival_rate',
                title='Survival Rate by Cancer Stage',
                labels={'cancer_stage': 'Cancer Stage', 'survival_rate': 'Survival Rate (%)'},
                color='survival_rate',
                color_continuous_scale='RdYlGn',
                text='survival_rate'
            )
            fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Chart 2: Distribution by Smoking Status
        if 'smoking_status' in df.columns:
            smoking_dist = df['smoking_status'].value_counts().reset_index()
            smoking_dist.columns = ['smoking_status', 'count']
            
            fig2 = px.pie(
                smoking_dist,
                values='count',
                names='smoking_status',
                title='Distribution of Patients by Smoking Status',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    # Row 2: Two more charts
    col3, col4 = st.columns(2)
    
    with col3:
        # Chart 3: Age Distribution
        if 'age' in df.columns:
            fig3 = px.histogram(
                df,
                x='age',
                nbins=30,
                title='Age Distribution of Patients',
                labels={'age': 'Age (years)', 'count': 'Number of Patients'},
                color_discrete_sequence=['#8b5cf6']
            )
            fig3.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # Chart 4: Sunburst - Survival by Stage and Treatment
        if 'cancer_stage' in df.columns and 'treatment_type' in df.columns:
            sunburst_data = df.copy()
            sunburst_data['survival_status'] = sunburst_data['survived'].map({0: 'Not Survived', 1: 'Survived'})
            
            fig4 = px.sunburst(
                sunburst_data,
                path=['cancer_stage', 'treatment_type', 'survival_status'],
                title='Survival Hierarchy: Stage → Treatment → Outcome',
                color='survived',
                color_continuous_scale='RdYlGn'
            )
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
    
    # Additional Analysis
    st.markdown("---")
    st.subheader("🔍 Additional Insights")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Gender distribution
        if 'gender' in df.columns:
            gender_survival = df.groupby('gender')['survived'].mean() * 100
            fig5 = px.bar(
                x=gender_survival.index,
                y=gender_survival.values,
                title='Survival Rate by Gender',
                labels={'x': 'Gender', 'y': 'Survival Rate (%)'},
                color=gender_survival.values,
                color_continuous_scale='Viridis'
            )
            fig5.update_layout(height=350)
            st.plotly_chart(fig5, use_container_width=True)
    
    with col6:
        # BMI vs Survival
        if 'bmi' in df.columns:
            fig6 = px.box(
                df,
                x='survived',
                y='bmi',
                title='BMI Distribution by Survival Status',
                labels={'survived': 'Survival Status', 'bmi': 'BMI'},
                color='survived',
                color_discrete_map={0: '#ef4444', 1: '#22c55e'}
            )
            fig6.update_xaxes(
    tickmode='array',
    tickvals=[0, 1],
    ticktext=['Not Survived', 'Survived']
)
            fig6.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig6, use_container_width=True)

# =====================================================
# PAGE 2: PREDICTION TOOL
# =====================================================
else:
    st.markdown('<p class="main-header">🤖 Patient Survival Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-powered prediction based on clinical features</p>',
        unsafe_allow_html=True
    )
    
    st.info("👨‍⚕️ Enter patient information below to predict survival probability")
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("📝 Patient Information")
        
        # Demographics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age (years)", 18, 100, 60)
            gender = st.selectbox("Gender", df['gender'].unique() if 'gender' in df.columns else ['Male', 'Female'])
        
        with col2:
            bmi = st.number_input("BMI", 15.0, 50.0, 25.0, 0.1)
            cholesterol = st.number_input("Cholesterol Level (mg/dL)", 100, 400, 200)
        
        with col3:
            country = st.selectbox("Country", sorted(df['country'].unique()) if 'country' in df.columns else ['USA'])
            smoking_status = st.selectbox(
                "Smoking Status",
                df['smoking_status'].unique() if 'smoking_status' in df.columns else ['Never', 'Former', 'Current']
            )
        
        st.markdown("---")
        
        # Clinical Information
        col4, col5, col6 = st.columns(3)
        
        with col4:
            cancer_stage = st.selectbox(
                "Cancer Stage",
                df['cancer_stage'].unique() if 'cancer_stage' in df.columns else ['I', 'II', 'III', 'IV']
            )
            treatment_type = st.selectbox(
                "Treatment Type",
                df['treatment_type'].unique() if 'treatment_type' in df.columns else ['Surgery', 'Chemotherapy', 'Radiation']
            )
        
        with col5:
            diagnosis_year = st.number_input("Diagnosis Year", 2010, 2024, 2020)
            diagnosis_month = st.selectbox("Diagnosis Month", list(range(1, 13)), index=0)
        
        with col6:
            # Additional features if they exist in the dataset
            additional_cols = [col for col in df.columns if col not in [
                'id', 'age', 'gender', 'bmi', 'cholesterol_level', 'country',
                'smoking_status', 'cancer_stage', 'treatment_type', 'survived',
                'diagnosis_date', 'end_treatment_date'
            ]]
            
            additional_inputs = {}
            for col in additional_cols[:3]:  # Limit to 3 additional features
                if df[col].dtype == 'object':
                    additional_inputs[col] = st.selectbox(f"{col.replace('_', ' ').title()}", df[col].unique())
                else:
                    additional_inputs[col] = st.number_input(
                        f"{col.replace('_', ' ').title()}",
                        float(df[col].min()),
                        float(df[col].max()),
                        float(df[col].median())
                    )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Survival", use_container_width=True)
    
    # Make prediction
    if submitted:
        # Create input dataframe
        input_data = {
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'cholesterol_level': cholesterol,
            'country': country,
            'smoking_status': smoking_status,
            'cancer_stage': cancer_stage,
            'treatment_type': treatment_type,
            'diagnosis_year': diagnosis_year,
            'diagnosis_month': diagnosis_month
        }
        
        # Add additional inputs
        input_data.update(additional_inputs)
        
        input_df = pd.DataFrame([input_data])
        
        try:
            # Preprocess and predict
            input_processed = preprocessor.transform(input_df)
            prediction = model.predict(input_processed)[0]
            prediction_proba = model.predict_proba(input_processed)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Create visualization
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if prediction == 1:
                    st.success("✅ **Prediction: SURVIVED**")
                    confidence = prediction_proba[1] * 100
                else:
                    st.error("❌ **Prediction: NOT SURVIVED**")
                    confidence = prediction_proba[0] * 100
                
                st.metric("Confidence Score", f"{confidence:.1f}%")
                
                # Risk assessment
                if confidence > 80:
                    risk_level = "Very High Confidence"
                    risk_color = "🟢"
                elif confidence > 60:
                    risk_level = "High Confidence"
                    risk_color = "🟡"
                else:
                    risk_level = "Moderate Confidence"
                    risk_color = "🟠"
                
                st.info(f"{risk_color} **{risk_level}**")
            
            with col2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction_proba[1] * 100,
                    title={'text': "Survival Probability"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "#fee2e2"},
                            {'range': [33, 66], 'color': "#fef3c7"},
                            {'range': [66, 100], 'color': "#d1fae5"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Probability breakdown
            st.markdown("---")
            st.subheader("📈 Probability Breakdown")
            
            prob_df = pd.DataFrame({
                'Outcome': ['Not Survived', 'Survived'],
                'Probability': [prediction_proba[0] * 100, prediction_proba[1] * 100]
            })
            
            fig_prob = px.bar(
                prob_df,
                x='Outcome',
                y='Probability',
                title='Prediction Probabilities',
                color='Outcome',
                color_discrete_map={'Not Survived': '#ef4444', 'Survived': '#22c55e'},
                text='Probability'
            )
            fig_prob.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_prob.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p>🎗️ Lung Cancer Survival Prediction System | Built with Streamlit & LightGBM</p>
        <p>⚠️ This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.</p>
    </div>
    """,
    unsafe_allow_html=True
)