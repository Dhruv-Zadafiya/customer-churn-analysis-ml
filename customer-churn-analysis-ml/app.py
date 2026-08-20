import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_data():

    file_path = BASE_DIR / "Customer_churn.csv"

    if not file_path.exists():
        st.error(
            f"Dataset not found: {file_path}"
        )
        st.stop()

    df = pd.read_csv(file_path)

    # -----------------------------------------------------
    # Convert Churn column into readable labels
    # -----------------------------------------------------

    if "Churn" not in df.columns:
        st.error(
            "The dataset must contain a 'Churn' column."
        )
        st.stop()

    if pd.api.types.is_numeric_dtype(df["Churn"]):

        df["Churn_Label"] = (
            df["Churn"]
            .map({
                0: "No",
                1: "Yes"
            })
            .fillna(df["Churn"].astype(str))
        )

    else:

        df["Churn_Label"] = (
            df["Churn"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        # Handle common text values
        df["Churn_Label"] = (
            df["Churn_Label"]
            .replace({
                "0": "No",
                "1": "Yes",
                "True": "Yes",
                "False": "No"
            })
        )

    return df


df = load_data()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive Customer Behavior, Churn Analysis & '
    'K-Means Segmentation Dashboard'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎯 Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🔴 Churn Analysis",
        "👥 Customer Explorer",
        "🔥 Correlation Analysis",
        "🤖 K-Means Segmentation",
        "📈 Cluster Profiles",
        "💡 Business Insights"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Built using Python, Pandas, NumPy, "
    "Matplotlib, Seaborn and Scikit-learn."
)


# =========================================================
# COMMON VALUES
# =========================================================

total_customers = len(df)

churned = (
    df["Churn_Label"] == "Yes"
).sum()

active = (
    df["Churn_Label"] == "No"
).sum()

churn_rate = (
    churned / total_customers * 100
    if total_customers > 0
    else 0
)


# =========================================================
# OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">Business Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "🔴 Churned Customers",
        f"{churned:,}"
    )

    col3.metric(
        "🟢 Active Customers",
        f"{active:,}"
    )

    col4.metric(
        "📉 Churn Rate",
        f"{churn_rate:.2f}%"
    )

    st.divider()

    # -----------------------------------------------------
    # QUICK STATISTICS
    # -----------------------------------------------------

    st.subheader("📌 Customer Statistics")

    c1, c2, c3, c4 = st.columns(4)

    if "Age" in df.columns:
        c1.metric(
            "Average Age",
            f"{df['Age'].mean():.1f}"
        )

    if "Tenure" in df.columns:
        c2.metric(
            "Average Tenure",
            f"{df['Tenure'].mean():.1f}"
        )

    if "Total Spend" in df.columns:
        c3.metric(
            "Average Total Spend",
            f"{df['Total Spend'].mean():.2f}"
        )

    if "Usage Frequency" in df.columns:
        c4.metric(
            "Average Usage Frequency",
            f"{df['Usage Frequency'].mean():.1f}"
        )

    st.divider()

    # -----------------------------------------------------
    # CHURN DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("📊 Churn Distribution")

    col1, col2 = st.columns(2)

    with col1:

        churn_data = (
            df["Churn_Label"]
            .value_counts()
            .reindex(["Yes", "No"])
            .fillna(0)
            .astype(int)
            .reset_index()
        )

        churn_data.columns = [
            "Churn",
            "Customers"
        ]

        st.bar_chart(
            churn_data.set_index("Churn")
        )

    with col2:

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        values = [churned, active]

        if sum(values) > 0:

            ax.pie(
                values,
                labels=["Churned", "Active"],
                autopct="%1.1f%%",
                startangle=90
            )

        ax.set_title(
            "Customer Churn Percentage"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    st.divider()

    # -----------------------------------------------------
    # DATASET PREVIEW
    # -----------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# CHURN ANALYSIS
# =========================================================

elif page == "🔴 Churn Analysis":

    st.header("🔴 Churn Analysis")

    # -----------------------------------------------------
    # GENDER FILTER
    # -----------------------------------------------------

    if "Gender" in df.columns:

        selected_gender = st.multiselect(
            "Select Gender",
            options=df["Gender"].dropna().unique(),
            default=df["Gender"].dropna().unique()
        )

        filtered = df[
            df["Gender"].isin(selected_gender)
        ]

    else:

        filtered = df.copy()

    # -----------------------------------------------------
    # CHURN BY GENDER
    # -----------------------------------------------------

    if "Gender" in filtered.columns:

        st.subheader("👤 Churn by Gender")

        gender_churn = pd.crosstab(
            filtered["Gender"],
            filtered["Churn_Label"]
        )

        st.bar_chart(
            gender_churn
        )

    # -----------------------------------------------------
    # TENURE VS CHURN
    # -----------------------------------------------------

    if "Tenure" in filtered.columns:

        st.subheader("📅 Tenure vs Churn")

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        sns.boxplot(
            data=filtered,
            x="Churn_Label",
            y="Tenure",
            ax=ax
        )

        ax.set_xlabel("Churn")
        ax.set_ylabel("Tenure")

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # -----------------------------------------------------
    # SUPPORT CALLS
    # -----------------------------------------------------

    if "Support Calls" in filtered.columns:

        st.subheader("☎️ Support Calls vs Churn")

        support_churn = (
            filtered
            .groupby("Churn_Label")["Support Calls"]
            .mean()
            .round(2)
        )

        st.bar_chart(
            support_churn
        )

    # -----------------------------------------------------
    # PAYMENT DELAY
    # -----------------------------------------------------

    if "Payment Delay" in filtered.columns:

        st.subheader("💳 Payment Delay vs Churn")

        payment_churn = (
            filtered
            .groupby("Churn_Label")["Payment Delay"]
            .mean()
            .round(2)
        )

        st.bar_chart(
            payment_churn
        )


# =========================================================
# CUSTOMER EXPLORER
# =========================================================

elif page == "👥 Customer Explorer":

    st.header("👥 Customer Explorer")

    st.write(
        "Use the filters below to explore customer records."
    )

    # -----------------------------------------------------
    # AGE & TENURE FILTERS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    if "Age" in df.columns:

        with col1:

            age_range = st.slider(
                "Age Range",
                int(df["Age"].min()),
                int(df["Age"].max()),
                (
                    int(df["Age"].min()),
                    int(df["Age"].max())
                )
            )

    else:

        age_range = (
            -np.inf,
            np.inf
        )

    if "Tenure" in df.columns:

        with col2:

            tenure_range = st.slider(
                "Tenure Range",
                int(df["Tenure"].min()),
                int(df["Tenure"].max()),
                (
                    int(df["Tenure"].min()),
                    int(df["Tenure"].max())
                )
            )

    else:

        tenure_range = (
            -np.inf,
            np.inf
        )

    # -----------------------------------------------------
    # GENDER
    # -----------------------------------------------------

    if "Gender" in df.columns:

        gender = st.multiselect(
            "Gender",
            df["Gender"].dropna().unique(),
            default=df["Gender"].dropna().unique()
        )

    else:

        gender = None

    # -----------------------------------------------------
    # SUBSCRIPTION
    # -----------------------------------------------------

    if "Subscription Type" in df.columns:

        subscription = st.multiselect(
            "Subscription Type",
            df["Subscription Type"].dropna().unique(),
            default=df["Subscription Type"].dropna().unique()
        )

    else:

        subscription = None

    # -----------------------------------------------------
    # CHURN
    # -----------------------------------------------------

    churn_filter = st.multiselect(
        "Churn",
        df["Churn_Label"].dropna().unique(),
        default=df["Churn_Label"].dropna().unique()
    )

    # -----------------------------------------------------
    # APPLY FILTERS
    # -----------------------------------------------------

    filtered_df = df[
        (df["Age"].between(*age_range))
        if "Age" in df.columns
        else True
    ]

    if "Tenure" in df.columns:

        filtered_df = filtered_df[
            filtered_df["Tenure"].between(
                *tenure_range
            )
        ]

    if gender is not None:

        filtered_df = filtered_df[
            filtered_df["Gender"].isin(gender)
        ]

    if subscription is not None:

        filtered_df = filtered_df[
            filtered_df["Subscription Type"].isin(
                subscription
            )
        ]

    filtered_df = filtered_df[
        filtered_df["Churn_Label"].isin(
            churn_filter
        )
    ]

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    st.subheader(
        f"🔎 {len(filtered_df):,} Customers Found"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv,
        file_name="filtered_customers.csv",
        mime="text/csv"
    )


# =========================================================
# CORRELATION ANALYSIS
# =========================================================

elif page == "🔥 Correlation Analysis":

    st.header("🔥 Correlation Analysis")

    numeric_columns = (
        df
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    selected_columns = st.multiselect(
        "Select Variables",
        numeric_columns,
        default=numeric_columns
    )

    if len(selected_columns) >= 2:

        correlation = (
            df[selected_columns]
            .corr()
        )

        fig, ax = plt.subplots(
            figsize=(11, 8)
        )

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax
        )

        ax.set_title(
            "Customer Feature Correlation Heatmap"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    else:

        st.warning(
            "Please select at least two variables."
        )


# =========================================================
# K-MEANS SEGMENTATION
# =========================================================

elif page == "🤖 K-Means Segmentation":

    st.header("🤖 Customer Segmentation")

    st.write(
        "Use K-Means clustering to identify groups of "
        "customers with similar characteristics."
    )

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    default_features = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction"
    ]

    available_features = [
        col
        for col in default_features
        if col in df.columns
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    selected_features = st.multiselect(
        "Select Features for Clustering",
        available_features,
        default=available_features
    )

    # -----------------------------------------------------
    # NUMBER OF CLUSTERS
    # -----------------------------------------------------

    k = st.slider(
        "Number of Clusters (K)",
        min_value=2,
        max_value=8,
        value=4
    )

    if len(selected_features) < 2:

        st.warning(
            "Select at least two numeric features."
        )

    else:

        X = df[
            selected_features
        ].copy()

        # -------------------------------------------------
        # HANDLE MISSING VALUES
        # -------------------------------------------------

        X = X.fillna(
            X.median()
        )

        # -------------------------------------------------
        # STANDARDIZATION
        # -------------------------------------------------

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X
        )

        # -------------------------------------------------
        # K-MEANS
        # -------------------------------------------------

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        clusters = kmeans.fit_predict(
            X_scaled
        )

        cluster_df = df.copy()

        cluster_df["Cluster"] = clusters

        # -------------------------------------------------
        # CLUSTER COUNTS
        # -------------------------------------------------

        st.subheader(
            "📊 Cluster Distribution"
        )

        cluster_counts = (
            cluster_df["Cluster"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            cluster_counts
        )

        # -------------------------------------------------
        # PCA
        # -------------------------------------------------

        st.subheader(
            "📍 PCA Cluster Visualization"
        )

        pca = PCA(
            n_components=2
        )

        components = pca.fit_transform(
            X_scaled
        )

        pca_df = pd.DataFrame(
            components,
            columns=[
                "Principal Component 1",
                "Principal Component 2"
            ]
        )

        pca_df["Cluster"] = (
            clusters.astype(str)
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        sns.scatterplot(
            data=pca_df,
            x="Principal Component 1",
            y="Principal Component 2",
            hue="Cluster",
            palette="viridis",
            alpha=0.65,
            s=45,
            ax=ax
        )

        ax.set_title(
            f"K-Means Clustering — K={k}"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # -------------------------------------------------
        # CLUSTER SUMMARY
        # -------------------------------------------------

        st.subheader(
            "📋 Cluster Profile"
        )

        summary_columns = [
            col
            for col in default_features
            if col in cluster_df.columns
            and pd.api.types.is_numeric_dtype(
                cluster_df[col]
            )
        ]

        cluster_summary = (
            cluster_df
            .groupby("Cluster")[summary_columns]
            .mean()
            .round(2)
        )

        st.dataframe(
            cluster_summary,
            use_container_width=True
        )

        # -------------------------------------------------
        # DOWNLOAD CLUSTERS
        # -------------------------------------------------

        cluster_csv = (
            cluster_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="⬇️ Download Customer Segments",
            data=cluster_csv,
            file_name="customer_segments.csv",
            mime="text/csv"
        )


# =========================================================
# CLUSTER PROFILES
# =========================================================

elif page == "📈 Cluster Profiles":

    st.header("📈 Customer Segment Profiles")

    features = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend"
    ]

    available_profile_features = [
        col
        for col in features
        if col in df.columns
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(available_profile_features) < 2:

        st.warning(
            "Not enough numeric features available "
            "for cluster profiling."
        )

    else:

        X = df[
            available_profile_features
        ].copy()

        X = X.fillna(
            X.median()
        )

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X
        )

        kmeans = KMeans(
            n_clusters=4,
            random_state=42,
            n_init=10
        )

        df_profile = df.copy()

        df_profile["Cluster"] = (
            kmeans.fit_predict(
                X_scaled
            )
        )

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        profile = (
            df_profile
            .groupby("Cluster")[
                available_profile_features
            ]
            .mean()
            .round(2)
        )

        st.subheader(
            "📋 Cluster Summary"
        )

        st.dataframe(
            profile,
            use_container_width=True
        )

        # -------------------------------------------------
        # TOTAL SPEND
        # -------------------------------------------------

        if "Total Spend" in df_profile.columns:

            st.subheader(
                "💰 Average Total Spend by Cluster"
            )

            spend = (
                df_profile
                .groupby("Cluster")[
                    "Total Spend"
                ]
                .mean()
                .round(2)
            )

            st.bar_chart(
                spend
            )

        # -------------------------------------------------
        # USAGE
        # -------------------------------------------------

        if "Usage Frequency" in df_profile.columns:

            st.subheader(
                "📱 Average Usage Frequency by Cluster"
            )

            usage = (
                df_profile
                .groupby("Cluster")[
                    "Usage Frequency"
                ]
                .mean()
                .round(2)
            )

            st.bar_chart(
                usage
            )

        # -------------------------------------------------
        # SUPPORT CALLS
        # -------------------------------------------------

        if "Support Calls" in df_profile.columns:

            st.subheader(
                "☎️ Average Support Calls by Cluster"
            )

            support = (
                df_profile
                .groupby("Cluster")[
                    "Support Calls"
                ]
                .mean()
                .round(2)
            )

            st.bar_chart(
                support
            )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

elif page == "💡 Business Insights":

    st.header("💡 Business Insights")

    st.subheader("🎯 Key Findings")

    st.markdown(
        """
        ### 1. Customer Churn

        Churn analysis helps identify customers who are leaving
        the business and supports targeted retention strategies.

        ### 2. Customer Segmentation

        K-Means clustering groups customers based on similarities
        in demographic and behavioral characteristics.

        ### 3. Premium Customers

        Customers with higher spending and frequent usage can be
        treated as valuable customer segments.

        ### 4. Retention Opportunities

        Customers with lower tenure and higher support activity
        can be investigated for potential retention campaigns.

        ### 5. Customer Engagement

        Usage frequency and total spending can help businesses
        understand customer engagement levels.

        ### 6. Data-Driven Decisions

        Customer segmentation allows businesses to design
        personalized marketing, engagement and retention strategies.
        """
    )

    st.success(
        "Customer analytics can help businesses understand "
        "customer behavior and improve retention strategies."
    )

    st.info(
        "This dashboard is designed for analytical exploration "
        "and does not represent a production churn prediction model."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Customer Churn Analysis using Machine Learning | "
    "Python • Pandas • NumPy • Seaborn • Scikit-learn • Streamlit"
)
