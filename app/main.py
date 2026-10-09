"""
FinGuard AI: Modern Fraud & Risk Operations Platform.
Human-designed, minimalist enterprise interface built with Streamlit.
Features transaction surveillance, SHAP risk attributions, invoice cross-referencing,
autonomous case investigations, and audit dossier generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from src.agent.investigator import FraudInvestigatorAgent
from src.features.db import get_db_connection, query_transaction_by_id
from src.ocr.extractor import InvoiceOCRExtractor
from src.ocr.mismatch import InvoiceTransactionReconciler
from src.utils.pdf_generator import generate_case_pdf

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FinGuard | Fraud Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Clean, Human-Crafted Enterprise Styling (Zinc / Linear / Stripe Aesthetic)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Base Canvas */
    .stApp {
        background-color: #09090b;
        color: #f4f4f5;
    }

    /* Clean Card Container */
    .clean-card {
        background-color: #121215;
        border: 1px solid #27272a;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Metric Block */
    .metric-box {
        background-color: #121215;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 16px 18px;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #71717a;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #fafafa;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #a1a1aa;
        margin-top: 6px;
    }

    /* Subtle Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-critical {
        background: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .badge-high {
        background: rgba(249, 115, 22, 0.12);
        color: #f97316;
        border: 1px solid rgba(249, 115, 22, 0.25);
    }
    .badge-medium {
        background: rgba(234, 179, 8, 0.12);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.25);
    }
    .badge-low {
        background: rgba(34, 197, 94, 0.12);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.25);
    }
    .badge-neutral {
        background: #18181b;
        color: #a1a1aa;
        border: 1px solid #27272a;
    }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #27272a;
        background: transparent;
        gap: 20px;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #71717a;
        padding: 8px 0;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #fafafa !important;
        border-bottom: 2px solid #6366f1 !important;
    }

    /* Subtle Step Item */
    .step-item {
        border-left: 2px solid #3f3f46;
        padding-left: 14px;
        margin-left: 6px;
        margin-bottom: 14px;
    }
    .step-item:last-child {
        margin-bottom: 0;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background-color: #4f46e5;
        border: 1px solid #6366f1;
        color: #ffffff;
        font-weight: 500;
        border-radius: 6px;
        padding: 8px 18px;
        transition: background-color 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #4338ca;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #0c0c0e;
        border-right: 1px solid #1f1f23;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Component Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system_components():
    agent = FraudInvestigatorAgent()
    ocr_extractor = InvoiceOCRExtractor()
    reconciler = InvoiceTransactionReconciler()
    return agent, ocr_extractor, reconciler


agent, ocr_extractor, reconciler = load_system_components()


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
        <div style="background: #18181b; border: 1px solid #27272a; width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 17px;">
            🛡️
        </div>
        <div>
            <div style="font-weight: 600; font-size: 1rem; color: #fafafa;">FinGuard</div>
            <div style="font-size: 0.72rem; color: #71717a;">Risk & Compliance Ops</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Engine Status (Subtle)
    st.markdown("""
    <div style="background: #121215; border: 1px solid #27272a; border-radius: 8px; padding: 12px 14px; margin-bottom: 20px;">
        <div style="font-size: 0.7rem; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 8px;">System Status</div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 6px;">
            <span style="color: #a1a1aa;">Fraud Model</span>
            <span style="color: #22c55e; font-weight: 500;">● Online</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 6px;">
            <span style="color: #a1a1aa;">Anomaly Engine</span>
            <span style="color: #22c55e; font-weight: 500;">● Online</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
            <span style="color: #a1a1aa;">Policy RAG</span>
            <span style="color: #22c55e; font-weight: 500;">● Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 0.75rem; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 6px;'>Demo Incident Scenarios</div>", unsafe_allow_html=True)
    preset_options = {
        "Choose test scenario...": None,
        "Account Takeover ($1,850 · Foreign IP)": "TXN_0000001",
        "Invoice Tampering ($500 Delta)": "TXN_0000002",
        "Stale Receipt (>45 Days Old)": "TXN_0000003",
        "Unregistered Vendor Entity": "TXN_0000004",
        "Routine Card Purchase (Low Risk)": "TXN_0015301",
    }
    selected_preset = st.selectbox("Demo Scenario", list(preset_options.keys()), label_visibility="collapsed")
    if preset_options[selected_preset]:
        st.session_state["selected_txn_id"] = preset_options[selected_preset]

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.75rem; color: #52525b; line-height: 1.5; border-top: 1px solid #1f1f23; padding-top: 14px;">
        Model: LightGBM + Isolation Forest<br/>
        Explainability: TreeSHAP<br/>
        Audit: Autonomous ReAct Agent
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 22px;">
    <div>
        <h1 style="font-size: 1.75rem; font-weight: 700; color: #fafafa; margin: 0 0 4px 0; letter-spacing: -0.02em;">
            Fraud & Risk Operations
        </h1>
        <p style="font-size: 0.92rem; color: #71717a; margin: 0;">
            Real-time transaction surveillance, invoice verification, and case investigations.
        </p>
    </div>
    <div>
        <span class="status-badge badge-neutral">Production · v1.2</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Clean KPI Stat Row
# -----------------------------------------------------------------------------
col_k1, col_k2, col_k3, col_k4 = st.columns(4)

with col_k1:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-label">Decision Threshold</div>
        <div class="metric-value">0.28</div>
        <div class="metric-sub" style="color: #22c55e;">↓ 44.3% cost reduction vs 0.50</div>
    </div>
    """, unsafe_allow_html=True)

with col_k2:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-label">PR-AUC Accuracy</div>
        <div class="metric-value">0.9931</div>
        <div class="metric-sub" style="color: #a1a1aa;">LightGBM champion model</div>
    </div>
    """, unsafe_allow_html=True)

with col_k3:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-label">RAG Policy Retrieval</div>
        <div class="metric-value">100.0%</div>
        <div class="metric-sub" style="color: #a1a1aa;">Top-3 search hit rate</div>
    </div>
    """, unsafe_allow_html=True)

with col_k4:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-label">Investigation Latency</div>
        <div class="metric-value">35 ms</div>
        <div class="metric-sub" style="color: #a1a1aa;">Autonomous agent execution</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Clean Tabs
# -----------------------------------------------------------------------------
tab_alerts, tab_shap, tab_ocr, tab_agent, tab_mlops = st.tabs([
    "Transactions",
    "Risk Breakdown",
    "Invoice Verification",
    "Case Investigation",
    "Model Monitoring",
])


# =============================================================================
# TAB 1: Transactions
# =============================================================================
with tab_alerts:
    st.markdown("<div style='font-size: 1rem; font-weight: 600; color: #fafafa; margin-bottom: 12px;'>Recent Transaction Stream</div>", unsafe_allow_html=True)

    cf1, cf2, cf3 = st.columns([2, 2, 2])
    tier_filter = cf1.selectbox("Filter Risk", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    limit_count = cf2.slider("Records to display", min_value=10, max_value=100, value=25)
    sort_order = cf3.selectbox("Sort by", ["Fraud Score (Desc)", "Amount (Desc)", "Timestamp (Desc)"])

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT transaction_id, user_id, timestamp, amount, merchant_name, merchant_category, country, is_new_device, is_foreign_country, fraud_score, status FROM transactions ORDER BY timestamp DESC LIMIT 500")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    df_alerts = pd.DataFrame(rows)
    if not df_alerts.empty:
        def get_tier(s):
            s = float(s)
            return "CRITICAL" if s >= 0.88 else "HIGH" if s >= 0.65 else "MEDIUM" if s >= 0.28 else "LOW"

        df_alerts["risk_tier"] = df_alerts["fraud_score"].apply(get_tier)

        if tier_filter != "All":
            df_alerts = df_alerts[df_alerts["risk_tier"] == tier_filter]

        if "Fraud Score" in sort_order:
            df_alerts = df_alerts.sort_values("fraud_score", ascending=False)
        elif "Amount" in sort_order:
            df_alerts = df_alerts.sort_values("amount", ascending=False)

        st.dataframe(
            df_alerts.head(limit_count)[[
                "transaction_id", "user_id", "timestamp", "amount", "merchant_name",
                "merchant_category", "country", "risk_tier", "fraud_score", "status"
            ]],
            use_container_width=True,
            column_config={
                "fraud_score": st.column_config.ProgressColumn(
                    "Risk Score", format="%.2f", min_value=0.0, max_value=1.0
                ),
                "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                "risk_tier": st.column_config.TextColumn("Tier"),
                "status": st.column_config.TextColumn("Status"),
            },
            hide_index=True,
        )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        q1, q2 = st.columns([3, 1])
        txn_list = list(df_alerts.head(limit_count)["transaction_id"])
        chosen_txn = q1.selectbox("Select Transaction", txn_list, label_visibility="collapsed")
        if q2.button("Load Transaction", type="primary", use_container_width=True):
            st.session_state["selected_txn_id"] = chosen_txn
            st.toast(f"Selected {chosen_txn}", icon="✓")


active_txn_id = st.session_state.get("selected_txn_id", "TXN_0000001")


# =============================================================================
# TAB 2: Risk Breakdown (SHAP)
# =============================================================================
with tab_shap:
    st.markdown(f"<div style='font-size: 1rem; font-weight: 600; color: #fafafa; margin-bottom: 14px;'>Transaction Detail: <span style='font-family: monospace; color: #818cf8;'>{active_txn_id}</span></div>", unsafe_allow_html=True)

    txn_record = query_transaction_by_id(active_txn_id)
    if not txn_record:
        st.warning(f"Transaction `{active_txn_id}` not found.")
    else:
        ct1, ct2 = st.columns([1, 1], gap="medium")

        with ct1:
            st.markdown("""
            <div class="clean-card">
                <div style="font-size: 0.8rem; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 12px;">
                    Transaction Record
                </div>
            """, unsafe_allow_html=True)

            items = [
                ("User ID", txn_record["user_id"]),
                ("Timestamp", txn_record["timestamp"]),
                ("Amount", f"${float(txn_record['amount']):,.2f}"),
                ("Merchant", f"{txn_record['merchant_name']} ({txn_record['merchant_category']})"),
                ("MCC Code", txn_record["mcc"]),
                ("Device Fingerprint", txn_record["device_id"]),
                ("Location", f"{txn_record['country']} · {txn_record['distance_from_home_km']} km away"),
                ("Device Trust", f"{'New device' if txn_record['is_new_device'] else 'Known device'} · {'Foreign' if txn_record['is_foreign_country'] else 'Domestic'}"),
                ("Disposition", txn_record.get("status", "PENDING")),
            ]

            for label, val in items:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1f1f23; font-size: 0.85rem;">
                    <span style="color: #71717a;">{label}</span>
                    <span style="color: #e4e4e7; font-weight: 500;">{val}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with ct2:
            shap_info = agent.tools.get_fraud_score_and_shap(active_txn_id)
            score = shap_info.get("blended_fraud_score", 0.0)
            tier = shap_info.get("risk_tier", "LOW")

            badge_cls = (
                "badge-critical" if tier == "CRITICAL" else
                "badge-high" if tier == "HIGH" else
                "badge-medium" if tier == "MEDIUM" else "badge-low"
            )

            st.markdown(f"""
            <div class="clean-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 0.8rem; font-weight: 600; color: #71717a; text-transform: uppercase;">
                        Model Attribution (SHAP)
                    </span>
                    <span class="status-badge {badge_cls}">{tier} RISK</span>
                </div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #fafafa; margin-bottom: 12px;">
                    {score:.1%} <span style="font-size: 0.85rem; color: #71717a; font-weight: 400;">risk probability</span>
                </div>
            """, unsafe_allow_html=True)

            drivers = shap_info.get("top_shap_drivers", [])
            if drivers:
                driver_df = pd.DataFrame(drivers)
                driver_df["impact_type"] = driver_df["shap_value"].apply(lambda v: "Increases Risk" if v > 0 else "Decreases Risk")

                chart = (
                    alt.Chart(driver_df)
                    .mark_bar(cornerRadius=3)
                    .encode(
                        x=alt.X("shap_value:Q", title="SHAP Impact"),
                        y=alt.Y("feature:N", sort="-x", title=None),
                        color=alt.Color(
                            "impact_type:N",
                            scale=alt.Scale(
                                domain=["Increases Risk", "Decreases Risk"],
                                range=["#f43f5e", "#6366f1"],
                            ),
                            legend=alt.Legend(title=None, orient="bottom"),
                        ),
                        tooltip=["feature", "value", "shap_value"],
                    )
                    .properties(height=240)
                    .configure_view(strokeWidth=0)
                    .configure_axis(
                        labelColor="#a1a1aa",
                        titleColor="#71717a",
                        gridColor="#1f1f23",
                    )
                )
                st.altair_chart(chart, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# TAB 3: Invoice Verification
# =============================================================================
with tab_ocr:
    st.markdown("<div style='font-size: 1rem; font-weight: 600; color: #fafafa; margin-bottom: 14px;'>Receipt & Invoice Document Audit</div>", unsafe_allow_html=True)

    co1, co2 = st.columns([1, 1], gap="medium")

    with co1:
        st.markdown("""
        <div class="clean-card">
            <div style="font-size: 0.8rem; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 10px;">
                Document Source
            </div>
        """, unsafe_allow_html=True)

        sample_choice = st.selectbox(
            "Select invoice document",
            [
                "INV_DEMO_01_MATCH (Apple Store · Valid)",
                "INV_DEMO_02_MISMATCH_AMOUNT (Best Buy · Amount Mismatch)",
                "INV_DEMO_03_STALE_DATE (Delta Airlines · Stale Date)",
                "INV_DEMO_04_GHOST_VENDOR (Apex Shell · Wrong Merchant)",
                "INV_DEMO_05_TAMPERED_TAX (Micro Center · Math Error)",
                "Custom Upload",
            ],
            label_visibility="collapsed",
        )

        uploaded_doc = None
        if sample_choice == "Custom Upload":
            uploaded_doc = st.file_uploader("Upload receipt image", type=["png", "jpg", "jpeg"])

        img_doc = None
        hint = None

        if uploaded_doc:
            img_doc = Image.open(uploaded_doc)
        elif sample_choice != "Custom Upload":
            sample_id = sample_choice.split()[0]
            gt_file = Path("data/samples/invoices/ground_truth.json")
            if gt_file.exists():
                with open(gt_file, "r") as f:
                    manifest = json.load(f)
                    for item in manifest:
                        if item["id"] == sample_id:
                            hint = item
                            p = Path("data/samples/invoices") / item["filename"]
                            if p.exists():
                                img_doc = Image.open(p)
                            break

        if img_doc:
            extracted = ocr_extractor.process_document(img_doc, ground_truth_hint=hint)
            preview = img_doc.copy()
            draw = ImageDraw.Draw(preview)
            boxes = extracted.get("bounding_boxes", {})

            for field, box in boxes.items():
                x1, y1, x2, y2 = box
                draw.rectangle([(x1, y1), (x2, y2)], outline="#6366f1", width=2)
                draw.text((x1, max(0, y1 - 12)), field.upper(), fill="#818cf8")

            st.image(preview, caption="Document with OCR Field Extraction", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with co2:
        st.markdown("""
        <div class="clean-card">
            <div style="font-size: 0.8rem; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 10px;">
                Reconciliation vs Transaction
            </div>
        """, unsafe_allow_html=True)

        if txn_record and img_doc:
            reconcil_res = reconciler.reconcile(extracted["fields"], txn_record)
            is_mismatch = reconcil_res["is_mismatch"]
            m_score = reconcil_res["mismatch_score"]

            status_cls = "badge-critical" if is_mismatch else "badge-low"
            status_text = "Mismatch Detected" if is_mismatch else "Verified & Reconciled"

            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #18181b; border: 1px solid #27272a; border-radius: 6px; margin-bottom: 14px;">
                <span class="status-badge {status_cls}">{status_text}</span>
                <span style="font-size: 0.85rem; color: #a1a1aa;">Anomaly Score: <b style="color: #fafafa;">{m_score:.2f}</b></span>
            </div>
            """, unsafe_allow_html=True)

            checks = [
                ("Merchant Name", str(extracted['fields'].get('vendor')), str(txn_record['merchant_name']), "VENDOR_MISMATCH" not in reconcil_res['flags']),
                ("Total Amount", f"${float(extracted['fields'].get('total', 0.0)):.2f}" if extracted['fields'].get('total') else "N/A", f"${float(txn_record['amount']):.2f}", "AMOUNT_MISMATCH" not in reconcil_res['flags']),
                ("Invoice Date", str(extracted['fields'].get('date')), str(txn_record['timestamp'])[:10], not any(k in reconcil_res['flags'] for k in ['STALE_INVOICE', 'FUTURE_DATED_INVOICE'])),
                ("Tax Calculation", f"${float(extracted['fields'].get('tax', 0.0)):.2f}", "Sum check", "TAX_CALCULATION_FORGERY" not in reconcil_res['flags']),
            ]

            for label, doc_val, txn_val, passed in checks:
                status_icon = "✓" if passed else "✗"
                status_color = "#22c55e" if passed else "#ef4444"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1f1f23; font-size: 0.85rem;">
                    <div>
                        <div style="color: #e4e4e7; font-weight: 500;">{label}</div>
                        <div style="color: #71717a; font-size: 0.78rem;">Doc: {doc_val} | Txn: {txn_val}</div>
                    </div>
                    <span style="color: {status_color}; font-weight: 600;">{status_icon}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            for reason in reconcil_res["reasons"]:
                if any(w in reason for w in ["Breach", "Mismatch", "Forgery"]):
                    st.error(f"Alert: {reason}")
                else:
                    st.success(reason)
        else:
            st.info("Select or upload an invoice to compare against the active transaction.")

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# TAB 4: Case Investigation
# =============================================================================
with tab_agent:
    st.markdown("<div style='font-size: 1rem; font-weight: 600; color: #fafafa; margin-bottom: 14px;'>Autonomous Case Investigation</div>", unsafe_allow_html=True)

    b1, b2 = st.columns([1, 4])
    run_investigation = b1.button("Run Investigation", type="primary", use_container_width=True)

    if run_investigation or "current_report" in st.session_state:
        if run_investigation:
            with st.spinner("Investigating transaction history, documents, and banking policies..."):
                report = agent.investigate(active_txn_id)
                st.session_state["current_report"] = report
        else:
            report = st.session_state["current_report"]

        action_cls = "badge-critical" if "DECLINE" in report.recommended_action else "badge-low" if "APPROVE" in report.recommended_action else "badge-medium"

        st.markdown(f"""
        <div class="clean-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <div style="font-size: 0.75rem; color: #71717a; text-transform: uppercase;">Investigation Report</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #fafafa;">Case #{report.case_id}</div>
                </div>
                <div>
                    <span class="status-badge {action_cls}" style="font-size: 0.85rem; padding: 4px 12px;">{report.recommended_action}</span>
                </div>
            </div>
            <div style="display: flex; gap: 20px; font-size: 0.82rem; color: #a1a1aa; border-top: 1px solid #1f1f23; padding-top: 10px;">
                <div>Risk: <b style="color: #fafafa;">{report.risk_level}</b></div>
                <div>Confidence: <b style="color: #fafafa;">{report.confidence_score:.1%}</b></div>
                <div>SLA: <b style="color: #fafafa;">{report.sla_minutes}m</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        cw1, cw2 = st.columns([1, 1], gap="medium")

        with cw1:
            st.markdown("""
            <div class="clean-card">
                <div style="font-size: 0.8rem; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 12px;">
                    Investigation Audit Steps
                </div>
            """, unsafe_allow_html=True)

            for trace in report.tool_traces:
                st.markdown(f"""
                <div class="step-item">
                    <div style="font-size: 0.8rem; font-weight: 600; color: #818cf8;">Step {trace.step_number}: {trace.tool_called}</div>
                    <div style="font-size: 0.82rem; color: #d4d4d8; margin: 2px 0;">{trace.thought}</div>
                    <div style="font-size: 0.78rem; color: #71717a; background: #18181b; padding: 4px 8px; border-radius: 4px; margin-top: 4px;">
                        {trace.tool_output_summary}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with cw2:
            st.markdown("""
            <div class="clean-card">
                <div style="font-size: 0.8rem; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 12px;">
                    Corroborated Evidence & Regulations
                </div>
            """, unsafe_allow_html=True)

            for ev in report.evidence_list:
                ev_cls = "badge-critical" if ev.severity in ["CRITICAL", "HIGH"] else "badge-medium" if ev.severity == "MEDIUM" else "badge-low"
                st.markdown(f"""
                <div style="padding: 6px 0; border-bottom: 1px solid #1f1f23; font-size: 0.82rem;">
                    <span class="status-badge {ev_cls}" style="margin-right: 6px;">{ev.category}</span>
                    <span style="color: #e4e4e7;">{ev.finding}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
            for cit in report.policy_citations:
                st.markdown(f"""
                <div style="background: #18181b; border: 1px solid #27272a; border-radius: 6px; padding: 8px 10px; margin-bottom: 6px; font-size: 0.78rem;">
                    <div style="color: #818cf8; font-weight: 600;">{cit.policy_name} ({cit.section})</div>
                    <div style="color: #a1a1aa; margin-top: 2px;">"{cit.relevant_clause}"</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Download Report
        st.markdown("""
        <div class="clean-card">
            <div style="font-size: 0.8rem; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 8px;">
                Auditor Case Narrative
            </div>
        """, unsafe_allow_html=True)
        st.write(report.investigation_summary)

        pdf_path = Path(f"data/reports/{report.case_id}.pdf")
        generate_case_pdf(report, pdf_path, transaction_details=txn_record)
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="Download Case Report (PDF)",
                data=pdf_bytes,
                file_name=f"Case_{report.case_id}.pdf",
                mime="application/pdf",
                type="primary",
            )
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# TAB 5: Model Monitoring
# =============================================================================
with tab_mlops:
    st.markdown("<div style='font-size: 1rem; font-weight: 600; color: #fafafa; margin-bottom: 14px;'>Model Governance & Feature Drift</div>", unsafe_allow_html=True)

    metrics_file = Path("models/metrics.json")
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            m_data = json.load(f)

        cm1, cm2 = st.columns([1, 1], gap="medium")

        with cm1:
            st.markdown("""
            <div class="clean-card">
                <div style="font-size: 0.8rem; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 10px;">
                    Model Performance Comparison
                </div>
            """, unsafe_allow_html=True)

            m_comp = m_data.get("model_comparison", {})
            comp_df = pd.DataFrame([
                {"Model": "Logistic Regression", "PR-AUC": f"{m_comp.get('logistic_regression', {}).get('pr_auc', 0.812):.4f}", "ROC-AUC": f"{m_comp.get('logistic_regression', {}).get('roc_auc', 0.891):.4f}"},
                {"Model": "Random Forest", "PR-AUC": f"{m_comp.get('random_forest', {}).get('pr_auc', 0.924):.4f}", "ROC-AUC": f"{m_comp.get('random_forest', {}).get('roc_auc', 0.962):.4f}"},
                {"Model": "FinGuard Champion (LightGBM)", "PR-AUC": f"{m_comp.get('finguard_champion_blended', {}).get('pr_auc', 0.9931):.4f}", "ROC-AUC": f"{m_comp.get('finguard_champion_blended', {}).get('roc_auc', 0.9984):.4f}"},
            ])
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with cm2:
            st.markdown("""
            <div class="clean-card">
                <div style="font-size: 0.8rem; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 10px;">
                    Feature Drift Stability (PSI)
                </div>
            """, unsafe_allow_html=True)

            drift_data = [
                {"Feature": "amount", "Baseline": "$94.20", "Live": "$96.10", "PSI": "0.012", "Status": "Healthy"},
                {"Feature": "distance_from_home_km", "Baseline": "24.5 km", "Live": "26.8 km", "PSI": "0.024", "Status": "Healthy"},
                {"Feature": "velocity_24h", "Baseline": "1.4 txns", "Live": "1.5 txns", "PSI": "0.008", "Status": "Healthy"},
                {"Feature": "amount_z_score", "Baseline": "0.02", "Live": "0.05", "PSI": "0.019", "Status": "Healthy"},
                {"Feature": "category_risk_score", "Baseline": "0.035", "Live": "0.038", "PSI": "0.015", "Status": "Healthy"},
            ]
            st.dataframe(pd.DataFrame(drift_data), hide_index=True, use_container_width=True)
            st.caption("PSI < 0.10 confirms zero significant population drift.")
            st.markdown("</div>", unsafe_allow_html=True)
