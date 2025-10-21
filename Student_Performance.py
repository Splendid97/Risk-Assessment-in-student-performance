import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="🎓 Student Performance Dashboard", layout="wide")

# -----------------------
# SIDEBAR NAVIGATION
# -----------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📈 Data Analysis", "🤖 Prediction System"])




# FILE UPLOAD

purple_palette = ["#7B61FF", "#B388FF", "#4C2BFF", "#9F7BFF", "#6F4BFF"]
file_path = r"C:\Users\Emmanuel\Downloads\student_performance_200.csv"
df = pd.read_csv(file_path)
st.sidebar.markdown("---")
st.sidebar.info("Developed by: *Splendid Emmanuel*")
st.sidebar.caption("SIWES Presentation Project 2025")


 # 🏠 HOME PAGE

if page == "🏠 Home":
        st.title("🎓 Student Performance Analysis Dashboard")
        st.markdown("This dashboard analyzes student performance data to help identify trends and detect at-risk students early.")

        total_students = len(df)
        avg_attendance = round(df["Attendance (%)"].mean(), 2)
        avg_exam = round(df["Exam Score"].mean(), 2)
        pass_rate = round(((df["Final Grade"].isin(["A", "B"])).mean()) * 100, 2)
        top_department = df.groupby("Department")["Exam Score"].mean().idxmax()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("👨‍🎓 Total Students", total_students)
        col2.metric("📚 Avg Attendance (%)", avg_attendance)
        col3.metric("🧮 Avg Exam Score", avg_exam)
        col4.metric("♻ Estimated Pass Rate", f"{pass_rate}%")
        col5.metric("🏅 Top Department", top_department)

        st.markdown("---")

        # Student by Department
        col1, col2 = st.columns(2)
        with col1:
            dept_count = df["Department"].value_counts().reset_index()
            dept_count.columns = ["Department", "Count"]
            fig1 = px.pie(dept_count, names="Department", values="Count",
                          title="Students by Department", hole=0.4)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            level_count = df["Level"].value_counts().reset_index()
            level_count.columns = ["Level", "Count"]
            fig2 = px.bar(level_count, x="Level", y="Count",
                          title="Number of Students by Level", text_auto=True)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 🎯 Insight Summary")
        st.write(f"""
        - **{total_students}** students analyzed across multiple departments.  
        - **Average attendance:** {avg_attendance}%  
        - **Average exam score:** {avg_exam}%  
        - **Pass rate:** {pass_rate}%  
        - **Top department:** {top_department}
        """)

    # -----------------------
    # 📈 DATA ANALYSIS PAGE
    # -----------------------
elif page == "📈 Data Analysis":
        st.title("📈 Data Analysis & Visualization")

        st.subheader("📊 Dataset Overview")
        st.dataframe(df.sample(10).reset_index(drop=True))

        st.subheader("📉 Summary Statistics")
        st.write(df.describe())

        st.markdown("---")
        st.subheader("Attendance vs Exam Score")
        fig_scatter = px.scatter(df, x="Attendance (%)", y="Exam Score", color="Department",
                                 hover_data=["Student_ID", "Final Grade"],
                                 title="Attendance vs Exam Score", trendline="ols")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        st.subheader("Average Scores by Department")
        df["Average Score"] = df[["Assignments Score", "Test Score", "Exam Score"]].mean(axis=1)
        dept_avg = df.groupby("Department")[["Assignments Score", "Test Score", "Exam Score", "Average Score"]].mean().reset_index()
        dept_avg = dept_avg.melt(id_vars="Department", var_name="Assessment", value_name="Score")
        fig_bar = px.bar(dept_avg, x="Department", y="Score", color="Assessment", barmode="group",
                         title="Average Assessment Scores by Department",
                         color_discrete_sequence=purple_palette)
        st.plotly_chart(fig_bar, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            dept_avg2 = df.groupby("Department")["Exam Score"].mean().reset_index()
            fig3 = px.bar(dept_avg2, x="Department", y="Exam Score",
                          title="Average Exam Score by Department", text_auto=True)
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            att_corr = df["Attendance (%)"].corr(df["Exam Score"])
            st.metric(label="Correlation between Attendance and Exam Score", value=round(att_corr, 2))
            fig4 = px.scatter(df, x="Attendance (%)", y="Exam Score", color="Department",
                              title="Attendance vs Exam Score")
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Grade Distribution")
        grade_count = df["Final Grade"].value_counts().reset_index()
        grade_count.columns = ["Final Grade", "Count"]
        fig5 = px.bar(grade_count, x="Final Grade", y="Count", color="Final Grade",
                      text_auto=True, title="Grade Distribution")
        st.plotly_chart(fig5, use_container_width=True)

    # 🤖 PREDICTION SYSTEM PAGE
    
elif page == "🤖 Prediction System":
        st.title("🤖 Predict At-Risk Students")
        st.write("This section classifies students as **At Risk** or **Pass Likely** using a rule-enhanced model.")

        # --- Rule-Based Logic + Random Forest ---
        df["Label"] = df["Final Grade"].map({"A": 1, "B": 1, "C": 0, "F": 0})
        features = ["Attendance (%)", "Assignments Score", "Test Score", "Exam Score"]
        X = df[features]
        y = df["Label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"✅ Model Accuracy: {round(acc * 100, 2)}%")

        # --- Fix: Prediction + Meaningful Status ---
        df["Predicted_Label"] = model.predict(X)
        df["Status"] = df.apply(
            lambda r: "At Risk" if (r["Attendance (%)"] < 60 or r["Exam Score"] < 50) else "Pass Likely",
            axis=1
        )

        # --- Color-coded table ---
        def color_status(val):
            color = "red" if val == "At Risk" else "limegreen"
            return f"color: {color}; font-weight: bold"

        styled_df = df.style.applymap(color_status, subset=["Status"])
        st.dataframe(styled_df)

        st.markdown("### 🎯 Prediction Summary")
        st.dataframe(df[["Student_ID", "Department", "Exam Score", "Final Grade", "Status"]].head(20))

        # --- Top At-Risk ---
        at_risk = df[df["Status"] == "At Risk"]
        st.markdown("### 🚨 Top At-Risk Students")
        st.dataframe(at_risk.sort_values(["Attendance (%)", "Exam Score"]).head(20)[
            ["Student_ID", "Department", "Level", "Attendance (%)", "Exam Score", "Final Grade", "Status"]
        ])

        # --- At-Risk by Department ---
        st.markdown("---")
        st.markdown("### 🏫 At-Risk by Department")
        ar_dept = at_risk["Department"].value_counts().reset_index()
        ar_dept.columns = ["Department", "Count"]
        if not ar_dept.empty:
            fig_ar = px.bar(ar_dept, x="Department", y="Count", title="At-Risk Students by Department",
                            color="Department", color_discrete_sequence=purple_palette)
            st.plotly_chart(fig_ar, use_container_width=True)
        else:
            st.info("✅ No at-risk students found.")

        # --- Download Results ---
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Prediction Results as CSV",
            data=csv,
            file_name="student_predictions.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Upload a student performance CSV file from the sidebar to begin analysis.")

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.markdown(
    f"<p style='text-align:center; color:gray;'>© {datetime.now().year} Akwa Ibom State University | Developed by: Udo Midighe-Abasi Emmanuel AK22/PHS/CSC/095</p>",
    unsafe_allow_html=True
)
