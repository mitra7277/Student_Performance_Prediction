# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_pdf, zip_pdfs  # Ensure utils.py exists

# -----------------------------
# FILE & FOLDER SETUP
# -----------------------------
USERS_FILE = "users.csv"
HISTORY_FILE = "user_history.csv"
PREDICTIONS_FILE = "predictions_record.csv"
FEEDBACK_FILE = "feedback_data.csv"
PDF_FOLDER = "batch_reports"
os.makedirs(PDF_FOLDER, exist_ok=True)

# -----------------------------
# SAFE CSV READ/WRITE HELPERS
# -----------------------------
def safe_read_csv(file, default_cols):
    if os.path.exists(file) and os.path.getsize(file) > 0:
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame(columns=default_cols)
    for col in default_cols:
        if col not in df.columns:
            df[col] = "" if col not in ["Predicted Score", "Rating"] else 0
    return df

def safe_save_csv(df, file):
    df.to_csv(file, index=False)

# -----------------------------
# USERS MANAGEMENT
# -----------------------------
def load_users():
    return safe_read_csv(USERS_FILE, ["username", "password"])

def register_user(username, password):
    df = load_users()
    if username in df["username"].values:
        return False
    new = pd.DataFrame([[username, password]], columns=["username", "password"])
    new.to_csv(USERS_FILE, mode='a', header=not os.path.exists(USERS_FILE) or os.path.getsize(USERS_FILE)==0, index=False)
    return True

def authenticate(username, password):
    df = load_users()
    return any((df["username"] == username) & (df["password"] == password))

# -----------------------------
# SESSION STATE INIT
# -----------------------------
for key in ["logged_in","username","last_prediction","user_predictions"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key=="user_predictions" else None if key=="last_prediction" else False if key=="logged_in" else ""

# -----------------------------
# LOGIN / SIGNUP
# -----------------------------
if not st.session_state.logged_in:
    st.title("🔐 Student Performance Prediction System")
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        login_user = st.text_input("Username")
        login_pass = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(login_user, login_pass):
                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.success(f"✅ Logged in as {login_user}")
                st.stop() 
            else:
                st.error("❌ Invalid username or password")

    with tab_signup:
        new_user = st.text_input("Choose a username", key="signup_user")
        new_pass = st.text_input("Choose a password", type="password", key="signup_pass")
        if st.button("Register"):
            if not new_user or not new_pass:
                st.warning("⚠️ Enter both username and password")
            else:
                ok = register_user(new_user, new_pass)
                if ok:
                    st.success("✅ Registration successful! You can now log in.")
                    st.balloons()
                else:
                    st.error("❌ Username already exists.")
    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.success(f"✅ Logged in as {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.stop() 

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = joblib.load("student_model_rf_8_features.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction System")
st.markdown("Predict students' final exam scores individually or in batch. View history, export PDFs, and collect feedback.")

# -----------------------------
# Tabs
# -----------------------------
tab_inputs, tab_predict, tab_charts, tab_feedback, tab_batch = st.tabs([
    "📥 Inputs", "🔮 Predict", "📊 Charts", "💬 Feedback", "📁 Batch"
])

# -----------------------------
# Inputs Tab
# -----------------------------
with tab_inputs:
    st.header("Enter Student Details (Individual)")
    c1, c2 = st.columns(2)
    with c1:
        hours = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        assignments = st.number_input("Assignments Submitted", 0, 50, 7)
        sleep = st.slider("Sleep Hours (per day)", 0, 12, 7)
    with c2:
        previous = st.number_input("Previous Score", 0, 100, 75)
        internet = st.slider("Internet Usage (hrs/day)", 0, 24, 3)
        study_env = st.selectbox("Study Environment", ["Home", "Library", "Cafe"])
        motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"])

# -----------------------------
# Predict Tab
# -----------------------------
# -----------------------------
# Tab: Predict (Interactive Charts)
# -----------------------------
with tab_predict:
    st.header("🔮 Predict Individual Score")
    
    if st.button("Predict Individual Score"):
        study_env_map = {"Home": 0, "Library": 1, "Cafe": 2}
        motivation_map = {"Low": 0, "Medium": 1, "High": 2}

        X = np.array([[hours, attendance, assignments, sleep, previous, internet,
                    study_env_map.get(study_env, 0), motivation_map.get(motivation, 1)]])
        try:
            pred = float(model.predict(X)[0])
            pred = max(0.0, min(100.0, pred))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            pred = None

        if pred is not None:
            st.metric("Predicted Score", f"{pred:.2f} / 100")
            st.progress(int(pred))

            # --- Save prediction in session state ---
            pred_entry = {
                "Username": st.session_state.username,
                "Hours Studied": hours,
                "Attendance": attendance,
                "Assignments": assignments,
                "Sleep Hours": sleep,
                "Previous Score": previous,
                "Internet Usage": internet,
                "Study Environment": study_env,
                "Motivation Level": motivation,
                "Predicted Score": pred,
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.last_prediction = pred_entry
            st.session_state.user_predictions.append(pred_entry)

            # --- Save to CSV files ---
            for file in [PREDICTIONS_FILE, HISTORY_FILE]:
                df_file = safe_read_csv(file, list(pred_entry.keys()))
                df_file = pd.concat([df_file, pd.DataFrame([pred_entry])], ignore_index=True)
                safe_save_csv(df_file, file)

            st.success("✅ Saved prediction to history and records.")

            # --- Generate PDF ---
            pdf_name = f"{st.session_state.username}_report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
            generate_pdf(pred_entry, pred, pdf_name)
            with open(pdf_name, "rb") as f:
                st.download_button("Download Student Report (PDF)", f, file_name=pdf_name, mime="application/pdf")

            # --- Mini Charts for Immediate Feedback ---
            st.subheader("📊 Instant Charts for This Prediction")

            # Line Chart: Last 5 Predictions
            df_user = pd.DataFrame(st.session_state.user_predictions)
            df_user["Date_parsed"] = pd.to_datetime(df_user["Date"], errors="coerce")
            df_user = df_user.sort_values("Date_parsed")
            fig_line = px.line(
                df_user.tail(5),
                x="Date_parsed",
                y="Predicted Score",
                markers=True,
                title="Last 5 Predictions"
            )
            fig_line.update_traces(line_color='blue', marker=dict(color='red', size=8))
            st.plotly_chart(fig_line, use_container_width=True)

            # Scatter: Hours Studied vs Predicted Score (Bubble = Attendance)
            fig_scatter = px.scatter(
                df_user.tail(5),
                x="Hours Studied",
                y="Predicted Score",
                size="Attendance",
                color="Motivation Level",
                color_discrete_map={"Low":"red", "Medium":"orange", "High":"green"},
                hover_data=["Assignments", "Sleep Hours", "Previous Score"],
                title="Hours Studied vs Predicted Score (Last 5)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Histogram: Score Distribution (Last 5)
            fig_hist = px.histogram(
                df_user.tail(5),
                x="Predicted Score",
                nbins=5,
                title="Distribution of Last 5 Predicted Scores",
                color_discrete_sequence=['green']
            )
            st.plotly_chart(fig_hist, use_container_width=True)


# -----------------------------
# Charts Tab
# -----------------------------

# Tab: Charts (User History)
# -----------------------------

# Tab: Charts & Analytics (User History + Advanced Visuals)
# -----------------------------
with tab_charts:
    st.header("📊 Your Predictions & Advanced Insights")

    history_cols = [
        "Username","Hours Studied","Attendance","Assignments","Sleep Hours",
        "Previous Score","Internet Usage","Study Environment","Motivation Level",
        "Predicted Score","Date"
    ]
    df_hist = safe_read_csv(HISTORY_FILE, history_cols)

    # Filter for the logged-in user
    user_df = df_hist[df_hist["Username"] == st.session_state.username]

    if not user_df.empty:
        st.subheader("Your Prediction History")
        st.dataframe(user_df.sort_values(by="Date", ascending=False).reset_index(drop=True))

        # ---------- Time-Series Line Chart ----------
        user_df["Date_parsed"] = pd.to_datetime(user_df["Date"], errors="coerce")
        user_df = user_df.sort_values("Date_parsed")
        fig_line = px.line(user_df, x="Date_parsed", y="Predicted Score",
                        markers=True, title="Your Predicted Scores Over Time")
        fig_line.update_traces(line_color='blue', marker=dict(color='red', size=8))
        st.plotly_chart(fig_line, use_container_width=True)

        # ---------- Histogram ----------
        fig_hist = px.histogram(user_df, x="Predicted Score", nbins=10,
                                title="Distribution of Your Predicted Scores",
                                color_discrete_sequence=['green'])
        st.plotly_chart(fig_hist, use_container_width=True)

        # ---------- Radar / Spider Chart ----------
        last_pred = user_df.iloc[-1]
        radar_data = [
            last_pred["Hours Studied"],
            last_pred["Attendance"],
            last_pred["Assignments"],
            last_pred["Sleep Hours"],
            last_pred["Internet Usage"],
            {"Home":0,"Library":1,"Cafe":2}.get(last_pred["Study Environment"],0),
            {"Low":0,"Medium":1,"High":2}.get(last_pred["Motivation Level"],1)
        ]
        radar_labels = ["Hours", "Attendance", "Assignments", "Sleep", "Internet", "Environment", "Motivation"]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_data,
            theta=radar_labels,
            fill='toself',
            name='Attributes'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(radar_data)+5])),
                                showlegend=True, title="Attribute Radar Chart")
        st.plotly_chart(fig_radar, use_container_width=True)

        # ---------- Gauge Chart for Last Predicted Score ----------
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=last_pred["Predicted Score"],
            title={'text': "Predicted Score"},
            gauge={'axis': {'range': [0, 100]},
                'bar': {'color': "blue"},
                'steps': [
                     {'range': [0, 50], 'color': "red"},
                       {'range': [50, 75], 'color': "yellow"},
                       {'range': [75, 100], 'color': "green"}]}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ---------- Correlation Heatmap ----------
        numeric_cols = ["Hours Studied", "Attendance", "Assignments", "Sleep Hours",
                        "Previous Score", "Internet Usage", "Predicted Score"]
        corr = user_df[numeric_cols].corr()
        fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis",
                             title="Correlation Heatmap of Attributes & Predicted Score")
        st.plotly_chart(fig_heat, use_container_width=True)

        # ---------- Box Plot ----------
        fig_box = px.box(user_df, y="Predicted Score", points="all", title="Box Plot of Predicted Scores")
        st.plotly_chart(fig_box, use_container_width=True)

        # ---------- Bubble Chart ----------
        motivation_map_inv = {0:"Low",1:"Medium",2:"High"}
        user_df["Motivation_Num"] = user_df["Motivation Level"].map({"Low":0,"Medium":1,"High":2})
        fig_bubble = px.scatter(user_df, x="Hours Studied", y="Predicted Score",
                                size="Attendance", color="Motivation_Num",
                                hover_data=["Assignments","Sleep Hours","Internet Usage"],
                                title="Hours vs Predicted Score (Bubble = Attendance, Color = Motivation)")
        st.plotly_chart(fig_bubble, use_container_width=True)

        # ---------- Top & Bottom Performance ----------
        df_sorted = user_df.sort_values("Predicted Score")
        top5 = df_sorted.tail(5)
        bottom5 = df_sorted.head(5)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=top5["Date_parsed"].dt.strftime("%Y-%m-%d"),
                                y=top5["Predicted Score"], name="Top 5 Scores", marker_color="green"))
        fig_bar.add_trace(go.Bar(x=bottom5["Date_parsed"].dt.strftime("%Y-%m-%d"),
                                y=bottom5["Predicted Score"], name="Bottom 5 Scores", marker_color="red"))
        fig_bar.update_layout(title="Top & Bottom Predicted Scores", barmode="group")
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("No predictions found. Make and save predictions to see your progress.")


# -----------------------------
# Feedback Tab
# -----------------------------
with tab_feedback:
    st.header("💬 Feedback & Ratings")
    feedback_text = st.text_area("Your feedback")
    rating = st.slider("Rate (1 = Poor, 5 = Excellent)", 1, 5, 3)
    if st.button("Submit Feedback"):
        entry = {
            "Username": st.session_state.username,
            "Feedback": feedback_text,
            "Rating": rating,
            "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        df_fb = safe_read_csv(FEEDBACK_FILE, list(entry.keys()))
        df_fb = pd.concat([df_fb, pd.DataFrame([entry])], ignore_index=True)
        safe_save_csv(df_fb, FEEDBACK_FILE)
        st.success("✅ Thank you for your feedback!")

# -----------------------------
# Batch Tab
# -----------------------------

# -----------------------------
# Tab: Batch Predictions (Teacher-Friendly)
# -----------------------------
with tab_batch:
    st.header("📁 Batch Predictions & Full Dashboard")
    uploaded = st.file_uploader("Upload CSV for batch predictions (8 features required)", type=["csv"])
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.dataframe(df_batch.head())

        # --- Mapping categorical ---
        study_env_map = {"Home":0,"Library":1,"Cafe":2}
        motivation_map = {"Low":0,"Medium":1,"High":2}
        df_batch["Study Environment"] = df_batch["Study Environment"].map(study_env_map).fillna(0).astype(int)
        df_batch["Motivation Level"] = df_batch["Motivation Level"].map(motivation_map).fillna(1).astype(int)

        required_cols = ["Hours Studied","Attendance","Assignments","Sleep Hours","Previous Score",
                    "Internet Usage","Study Environment","Motivation Level"]
        if all(col in df_batch.columns for col in required_cols):
            # --- Predict ---
            X_batch = df_batch[required_cols].values
            preds = model.predict(X_batch).clip(0,100)
            df_batch["Predicted Score"] = preds
            df_batch["Username"] = st.session_state.username
            st.success("✅ Batch predictions generated")
            st.dataframe(df_batch.head(15))

            # --- Average Score by Motivation Level ---
            df_batch["Motivation_Label"] = df_batch["Motivation Level"].map({0:"Low",1:"Medium",2:"High"})
            avg_scores = df_batch.groupby("Motivation_Label")["Predicted Score"].mean().reset_index()
            fig_avg = px.bar(
                avg_scores,
                x="Motivation_Label",
                y="Predicted Score",
                title="📊 Average Predicted Score by Motivation Level",
                color="Motivation_Label",
                color_discrete_map={"Low":"red","Medium":"orange","High":"green"},
                text_auto=".2f"
            )
            st.plotly_chart(fig_avg, use_container_width=True)

            # --- Top 5 & Bottom 5 Students ---
            st.subheader("🏆 Top 5 Students by Predicted Score")
            st.dataframe(df_batch.nlargest(5, "Predicted Score")[["Hours Studied","Attendance","Predicted Score","Motivation_Label"]])
            st.subheader("⚠️ Bottom 5 Students by Predicted Score")
            st.dataframe(df_batch.nsmallest(5, "Predicted Score")[["Hours Studied","Attendance","Predicted Score","Motivation_Label"]])

            # --- Histogram: Score Distribution ---
            fig_hist_batch = px.histogram(
                df_batch,
                x="Predicted Score",
                nbins=12,
                title="🟢 Distribution of Predicted Scores",
                color="Motivation_Label",
                color_discrete_map={"Low":"red","Medium":"orange","High":"green"}
            )
            st.plotly_chart(fig_hist_batch, use_container_width=True)

            # --- Scatter: Hours Studied vs Predicted Score ---
            fig_scatter_batch = px.scatter(
                df_batch,
                x="Hours Studied",
                y="Predicted Score",
                size="Attendance",
                color="Motivation_Label",
                title="⏱ Hours Studied vs Predicted Score (Bubble = Attendance)",
                color_discrete_map={"Low":"red","Medium":"orange","High":"green"},
                hover_data=["Assignments","Previous Score"]
            )
            st.plotly_chart(fig_scatter_batch, use_container_width=True)

        else:
            st.error(f"Missing columns! Required: {required_cols}")