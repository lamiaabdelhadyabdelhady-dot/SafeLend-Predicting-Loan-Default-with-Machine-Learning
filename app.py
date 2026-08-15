import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SafeLend - Cyber Credit Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",)


st.markdown(
    """
    <style>
    /* Dark Deep Navy/Black Background */
    .stApp {
        background-color: #080B10;
        color: #FFFFFF;
    }

    [data-testid="stHeader"], header {
        background-color: transparent !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F141C !important;
        border-right: 2px solid #1E2638 !important;
    }

    /* All Labels White & Crisp */
    label, p, span, h1, h2, h3, h4, .stMarkdown {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Hero Banner Styling */
    .hero-banner {
        background: linear-gradient(135deg, #111827 0%, #1E1B4B 100%);
        border: 2px solid #06B6D4;
        border-radius: 20px;
        padding: 25px 35px;
        box-shadow: 0 0 30px rgba(6, 182, 212, 0.25);
        margin-bottom: 25px;
    }

    /* 🔥 NEW MATRIX / METRIC CARDS DESIGN (تنسيق الماتريكس الجديد بالكامل) */
    .matrix-card {
        background: linear-gradient(145deg, #111827 0%, #0F172A 100%);
        border: 1px solid #1E293B;
        border-top: 4px solid #06B6D4; /* خط علوي ملون متوهج */
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    .matrix-card:hover {
        transform: translateY(-5px);
        border-top-color: #A855F7; /* يتغير للبنفسجي المضيء عند مرور الماوس */
        box-shadow: 0 15px 30px rgba(168, 85, 247, 0.3);
    }

    .matrix-title {
        color: #94A3B8 !important;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .matrix-value {
        color: #06B6D4 !important; /* لون تركواز الماتريكس */
        font-size: 2.2rem;
        font-weight: 900;
        margin: 5px 0;
    }
    .matrix-sub {
        font-size: 0.85rem;
        color: #38BDF8 !important;
        font-weight: 600;
    }

    /* Electric Neon Action Button */
    div.stButton > button {
        background: linear-gradient(90deg, #06B6D4 0%, #3B82F6 50%, #8B5CF6 100%);
        color: #FFFFFF !important;
        border: none;
        border-radius: 14px;
        padding: 22px 30px !important;
        font-weight: 900 !important;
        font-size: 22px !important;
        letter-spacing: 1px;
        box-shadow: 0 0 25px rgba(6, 182, 212, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 40px rgba(139, 92, 246, 0.7);
    }

    /* Tabs Custom Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border-radius: 10px;
        color: #FFFFFF !important;
        padding: 12px 25px;
        font-weight: bold;
        border: 1px solid #1E293B;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #06B6D4 0%, #3B82F6 100%) !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,)


@st.cache_resource
def load_models():
    preprocessor = joblib.load("preprocessor.joblib")
    selector = joblib.load("selector.joblib")
    scaler = joblib.load("scaler.joblib")
    rf_model = joblib.load("rf_model.joblib")
    return preprocessor, selector, scaler, rf_model


preprocessor, selector, scaler, rf_model = load_models()

st.markdown(
    """
    <div class="hero-banner">
        <h1 style="font-size: 2.8rem; margin:0; background: linear-gradient(90deg, #06B6D4, #A855F7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🛡️ SafeLend Cyber Assessment
        </h1>
        <p style="font-size: 1.1rem; color: #94A3B8 !important; margin-top:5px;">
            AI-Driven Credit Risk Intelligence & Automated Underwriting System
        </p>
    </div>
    """,
    unsafe_allow_html=True,)

st.sidebar.markdown(
    "<h2 style='color: #06B6D4 !important;'>📋 Applicant Data</h2>",
    unsafe_allow_html=True,)
person_age = st.sidebar.slider("Age (Years)", 18, 100, 28)
person_income = st.sidebar.number_input(
    "Annual Income ($)", min_value=0, max_value=1000000, value=75000, step=1000
)
person_emp_length = st.sidebar.number_input(
    "Employment Length (Years)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.5,
)
person_home_ownership = st.sidebar.selectbox(
    "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

st.sidebar.markdown(
    "<h2 style='color: #06B6D4 !important;'>💰 Loan Details</h2>",
    unsafe_allow_html=True,
)
loan_intent = st.sidebar.selectbox(
    "Loan Purpose",
    [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ],
)
loan_grade = st.sidebar.select_slider(
    "Credit Grade", options=["A", "B", "C", "D", "E", "F", "G"], value="B"
)
loan_amnt = st.sidebar.number_input(
    "Loan Amount ($)", min_value=500, max_value=50000, value=15000, step=500
)
loan_int_rate = st.sidebar.slider(
    "Interest Rate (%)", 1.0, 35.0, 11.5, step=0.1
)

st.sidebar.markdown(
    "<h2 style='color: #06B6D4 !important;'>📜 Credit History</h2>",
    unsafe_allow_html=True,
)
cb_person_default_on_file = st.sidebar.radio(
    "Historical Default?", ["N", "Y"], horizontal=True
)
cb_person_cred_hist_length = st.sidebar.slider(
    "Credit History (Years)", 0, 30, 4
)

loan_percent_income = (
    round(loan_amnt / person_income, 2) if person_income > 0 else 0.0
)

# 6. Navigation Tabs
tab1, tab2, tab3 = st.tabs(
    ["🎯 Risk Assessment", "📊 Loan Analytics", "ℹ️ Bank Guidelines"]
)

with tab1:
    st.write(" ")
    # 🔥 METRIC MATRIX DESIGN (تصميم الماتريكس ببطاقات مخصصة)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(
            f"""
            <div class="matrix-card">
                <div class="matrix-title">👤 Applicant Age</div>
                <div class="matrix-value">{person_age} <span style="font-size:1.2rem; color:#94A3B8;">Yrs</span></div>
                <div class="matrix-sub">Verified Profile</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="matrix-card">
                <div class="matrix-title">💵 Annual Income</div>
                <div class="matrix-value">${person_income:,.0f}</div>
                <div class="matrix-sub">Declared Revenue</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="matrix-card">
                <div class="matrix-title">🏦 Loan Amount</div>
                <div class="matrix-value">${loan_amnt:,.0f}</div>
                <div class="matrix-sub">Principal Requested</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        ratio_color = "#EF4444" if loan_percent_income > 0.4 else "#10B981"
        st.markdown(
            f"""
            <div class="matrix-card" style="border-top-color: {ratio_color};">
                <div class="matrix-title">📊 Debt-To-Income</div>
                <div class="matrix-value" style="color: {ratio_color} !important;">{loan_percent_income * 100:.1f}%</div>
                <div class="matrix-sub" style="color: {ratio_color} !important;">{"High Exposure" if loan_percent_income > 0.4 else "Healthy Threshold"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write(" ")
    st.write(" ")

    if st.button("🚀 EXECUTE AI RISK ASSESSMENT", use_container_width=True):
        input_df = pd.DataFrame(
            [
                {
                    "person_age": person_age,
                    "person_income": person_income,
                    "person_home_ownership": person_home_ownership,
                    "person_emp_length": person_emp_length,
                    "loan_intent": loan_intent,
                    "loan_grade": loan_grade,
                    "loan_amnt": loan_amnt,
                    "loan_int_rate": loan_int_rate,
                    "loan_percent_income": loan_percent_income,
                    "cb_person_default_on_file": cb_person_default_on_file,
                    "cb_person_cred_hist_length": cb_person_cred_hist_length,
                }
            ]
        )

        encoded = preprocessor.transform(input_df)
        selected = selector.transform(encoded)
        scaled = scaler.transform(selected)

        prediction = rf_model.predict(scaled)[0]
        probability = rf_model.predict_proba(scaled)[0][1]

        st.write(" ")
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.subheader("🎯 Evaluation Result")
            if prediction == 1:
                st.error("### ❌ HIGH RISK - REJECTED")
                st.write(
                    "High probability of financial default. Risk parameters exceed safety thresholds."
                )
            else:
                st.success("### ✅ LOW RISK - APPROVED")
                st.write(
                    "Credit profile is solid. Approved for immediate loan processing."
                )

        with res_col2:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={"text": "Default Probability Score (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#06B6D4"},
                        "steps": [
                            {
                                "range": [0, 35],
                                "color": "rgba(16, 185, 129, 0.4)",
                            },
                            {
                                "range": [35, 60],
                                "color": "rgba(245, 158, 11, 0.4)",
                            },
                            {
                                "range": [60, 100],
                                "color": "rgba(239, 68, 68, 0.4)",
                            },
                        ],
                    },
                )
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                height=250,
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📈 Interactive Loan Distribution")
    dummy_data = pd.DataFrame(
        {
            "Loan Amount": [
                5000,
                10000,
                15000,
                20000,
                25000,
                30000,
                35000,
            ],
            "Frequency": [120, 340, 800, 450, 200, 90, 30],
        }
    )

    fig_hist = px.bar(
        dummy_data,
        x="Loan Amount",
        y="Frequency",
        title="Approved Loans Benchmark",
        color_discrete_sequence=["#06B6D4"],
    )
    fig_hist.add_vline(
        x=loan_amnt,
        line_dash="dash",
        line_color="#A855F7",
        annotation_text="Applicant Request",
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.subheader("📜 SafeLend Underwriting Policies")
    st.markdown(
        """
    * **Credit Grades A & B:** Premier clients eligible for lowest interest rates.
    * **Debt-To-Income Threshold:** Max recommended limit is **40%**.
    * **Historical Default:** Previous defaults trigger an automatic full audit protocol.
    """
    )
