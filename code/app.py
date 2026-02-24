"""
HealthGuard AI – Streamlit Application
Multi-agent clinical decision support system.
"""

import os
import sys
import json
import time

import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.corpus import chunk_corpus, build_faiss_index
from orchestrator import run_healthguard
from agents.hospital_finder import recommend_hospitals, geocode_address, fetch_location_suggestions

try:
    from streamlit_js_eval import streamlit_js_eval
    HAS_JS_EVAL = True
except ImportError:
    HAS_JS_EVAL = False

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a5276 0%, #2e86c1 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p { color: #d5e8f0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Agent step cards */
    .agent-card {
        background: #f8f9fa;
        border-left: 4px solid #2e86c1;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.6rem;
    }
    .agent-card.active {
        border-left-color: #f39c12;
        background: #fef9e7;
    }
    .agent-card.done {
        border-left-color: #27ae60;
        background: #eafaf1;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fdedec;
        border: 1px solid #e74c3c;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #922b21;
        margin-bottom: 1rem;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: #eaf2f8;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        flex: 1;
        text-align: center;
    }
    .metric-box .label { font-size: 0.75rem; color: #5d6d7e; text-transform: uppercase; }
    .metric-box .value { font-size: 1.4rem; font-weight: 700; color: #1a5276; }

    /* Hide default streamlit footer */
    footer { visibility: hidden; }

    /* Hospital recommendation section */
    .hospital-header {
        background: linear-gradient(135deg, #1a8754 0%, #27ae60 100%);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        color: white;
        margin: 1.2rem 0 0.8rem 0;
    }
    .hospital-header h3 { color: white; margin: 0; font-size: 1.2rem; }
    .hospital-header p { color: #d5f5e3; margin: 0.2rem 0 0 0; font-size: 0.85rem; }

    .urgency-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .urgency-emergency { background: #e74c3c; color: white; }
    .urgency-urgent { background: #f39c12; color: white; }
    .urgency-routine { background: #2ecc71; color: white; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>HealthGuard AI</h1>
    <p>Multi-agent clinical decision support &mdash; differential diagnosis with traceable citations</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    <strong>Disclaimer:</strong> This tool is a <em>prototype for educational and demonstration purposes only</em>.
    It is NOT a certified medical device and must NOT be used for actual clinical decision-making.
    Always consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "OpenAI API Key (optional)",
        type="password",
        help="If provided, the system uses GPT models for extraction and diagnosis. "
             "Otherwise, a built-in rule-based engine is used.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    top_k = st.slider("Literature chunks to retrieve", 3, 15, 8)

    st.divider()
    st.header("Architecture")
    st.markdown("""
    **Agent Pipeline:**
    1. **Symptom Extractor** – NLP extraction of clinical findings
    2. **Literature Retriever** – Semantic search over medical corpus (FAISS + MiniLM)
    3. **Differential Generator** – Ranked diagnosis with evidence
    4. **Output Validator** – Citation integrity check

    **Stack:** Python, Streamlit, FAISS, sentence-transformers, OpenAI (optional)
    """)

    st.divider()
    st.caption("HealthGuard AI MVP | For demonstration purposes only")

    st.divider()
    st.header("Location Settings")

    # Initialize session state for location
    if "user_lat" not in st.session_state:
        st.session_state.user_lat = None
        st.session_state.user_lon = None
        st.session_state.location_source = None

    # Location autocomplete state
    if "loc_suggestions" not in st.session_state:
        st.session_state.loc_suggestions = []
    if "loc_query_last" not in st.session_state:
        st.session_state.loc_query_last = ""
    if "loc_selected_label" not in st.session_state:
        st.session_state.loc_selected_label = None

    hospital_radius = st.slider(
        "Hospital search radius (km)", 5, 50, 10,
        help="Search radius for nearby hospitals after diagnosis.",
    )

    # --- Location Autocomplete ---
    location_query = st.text_input(
        "Search for your location",
        placeholder="e.g., Imphal, Boston, 123 Main St...",
        help="Type at least 3 characters to see place suggestions.",
        key="loc_search_input",
    )

    current_query = location_query.strip()
    if current_query and len(current_query) >= 3 and current_query != st.session_state.loc_query_last:
        st.session_state.loc_suggestions = fetch_location_suggestions(current_query, limit=5)
        st.session_state.loc_query_last = current_query
        st.session_state.loc_selected_label = None
        st.session_state.user_lat = None
        st.session_state.user_lon = None
        st.session_state.location_source = None
    elif not current_query:
        st.session_state.loc_suggestions = []
        st.session_state.loc_query_last = ""
        st.session_state.loc_selected_label = None

    suggestions = st.session_state.loc_suggestions
    if suggestions:
        suggestion_labels = [s["label"] for s in suggestions]

        default_index = 0
        if st.session_state.loc_selected_label in suggestion_labels:
            default_index = suggestion_labels.index(st.session_state.loc_selected_label)

        selected_label = st.selectbox(
            "Select a location",
            options=suggestion_labels,
            index=default_index,
            key="loc_suggestion_select",
        )

        if selected_label:
            selected_suggestion = next(
                (s for s in suggestions if s["label"] == selected_label),
                None,
            )
            if selected_suggestion:
                st.session_state.loc_selected_label = selected_label
                st.session_state.user_lat = selected_suggestion["lat"]
                st.session_state.user_lon = selected_suggestion["lon"]
                st.session_state.location_source = "autocomplete"
                st.success(
                    f"Location set: {selected_suggestion['lat']:.4f}, "
                    f"{selected_suggestion['lon']:.4f}"
                )
    elif current_query and len(current_query) >= 3:
        st.warning("No suggestions found. Trying direct geocoding...")
        coords = geocode_address(current_query)
        if coords:
            st.session_state.user_lat, st.session_state.user_lon = coords
            st.session_state.location_source = "address"
            st.success(f"Location set: {coords[0]:.4f}, {coords[1]:.4f}")
        else:
            st.error("Could not find that location. Try a different search term.")
    elif current_query and len(current_query) < 3:
        st.caption("Type at least 3 characters to search.")


# ---------------------------------------------------------------------------
# Load / cache the embedding model and index
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Building medical literature index...")
def load_index(_model):
    chunks = chunk_corpus(chunk_size=500, chunk_overlap=100)
    index, chunk_list, _ = build_faiss_index(chunks, _model)
    return index, chunk_list


model = load_model()
faiss_index, chunk_list = load_index(model)


# ---------------------------------------------------------------------------
# Example vignettes
# ---------------------------------------------------------------------------
EXAMPLES = {
    "Select an example...": "",
    "Pneumonia presentation": (
        "45-year-old male presents with 3 days of productive cough with yellow sputum, "
        "fever (38.9°C), pleuritic chest pain on the right side, and shortness of breath. "
        "He is a current smoker with 20 pack-year history. Vitals: HR 102, RR 24, BP 130/85, "
        "SpO2 93% on room air. Exam reveals crackles in the right lower lobe. "
        "Chest X-ray shows right lower lobe consolidation."
    ),
    "Heart failure exacerbation": (
        "72-year-old female with known history of heart failure (EF 30%) and diabetes presents "
        "with worsening dyspnea on exertion over the past week, now experiencing orthopnea "
        "and paroxysmal nocturnal dyspnea. She reports weight gain of 4 kg in 5 days and "
        "bilateral lower extremity edema. Vitals: HR 98, RR 22, BP 100/65, SpO2 91%. "
        "Exam reveals elevated JVP, S3 gallop, bilateral crackles, and 2+ pitting edema. "
        "BNP 1850 pg/mL."
    ),
    "Chest pain – ACS concern": (
        "58-year-old male with hypertension and diabetes presents to the ED with sudden-onset "
        "crushing substernal chest pain radiating to the left arm, associated with diaphoresis "
        "and nausea for the past 2 hours. He is a former smoker. Vitals: HR 110, RR 20, "
        "BP 155/95, SpO2 97%. ECG shows ST elevation in leads V1-V4. "
        "Initial troponin I elevated at 2.5 ng/mL."
    ),
    "Dyspnea – multiple possibilities": (
        "65-year-old obese female presents with acute onset dyspnea and pleuritic chest pain. "
        "She had right knee replacement surgery 10 days ago and has been relatively immobile. "
        "She also reports mild cough without sputum and one episode of hemoptysis. "
        "Vitals: HR 115, RR 28, BP 138/88, SpO2 89% on room air. Exam reveals tachycardia, "
        "clear lung fields, and mild right calf swelling. D-dimer is 3200 ng/mL."
    ),
    "Fever with multiple symptoms": (
        "35-year-old male healthcare worker presents with 5 days of high fever (39.5°C), "
        "dry cough, severe fatigue, myalgia, headache, and loss of smell and taste. "
        "He reports progressive shortness of breath over the last 24 hours. "
        "No significant past medical history. Vitals: HR 100, RR 24, BP 125/80, "
        "SpO2 92% on room air. Chest X-ray shows bilateral ground-glass opacities."
    ),
}


# ---------------------------------------------------------------------------
# Main input
# ---------------------------------------------------------------------------
col_input, col_output = st.columns([1, 1.4], gap="large")

with col_input:
    st.subheader("Clinical Notes Input")

    example = st.selectbox("Quick examples", list(EXAMPLES.keys()))
    default_text = EXAMPLES.get(example, "")

    notes = st.text_area(
        "Enter clinical notes",
        value=default_text,
        height=250,
        placeholder="e.g., 45-year-old male with fever, cough, and shortness of breath...",
    )

    run_btn = st.button("Analyze", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
with col_output:
    st.subheader("Diagnostic Output")

    if run_btn and notes.strip():
        # Progress indicators
        status_container = st.container()

        with st.spinner("Running HealthGuard AI pipeline..."):
            result = run_healthguard(
                notes=notes,
                embedding_model=model,
                faiss_index=faiss_index,
                chunks=chunk_list,
                top_k=top_k,
            )

        # Store result in session state so it survives reruns
        st.session_state.pipeline_result = result
        st.session_state.pipeline_notes = notes

    # Display results from session state (persists across reruns)
    result = st.session_state.get("pipeline_result")

    if result and result.error:
        st.error(result.error)
    elif result:
        # Timing metrics
        total_time = sum(result.timings.values())
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="label">Findings</div>
                <div class="value">{len(result.findings)}</div>
            </div>
            <div class="metric-box">
                <div class="label">Chunks Retrieved</div>
                <div class="value">{len(result.literature)}</div>
            </div>
            <div class="metric-box">
                <div class="label">Citations Valid</div>
                <div class="value">{"Yes" if result.validation.get("valid", False) else "Review"}</div>
            </div>
            <div class="metric-box">
                <div class="label">Total Time</div>
                <div class="value">{total_time:.1f}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main differential
        st.markdown(result.differential)

        # Validation warnings
        if not result.validation.get("valid", True) and result.validation.get("issues"):
            with st.expander("Validation Notes", expanded=False):
                for issue in result.validation["issues"]:
                    st.warning(issue)

        # Extracted findings
        with st.expander("Extracted Clinical Findings", expanded=False):
            for f in result.findings:
                val = f" = {f['value']}" if f.get("value") else ""
                ctx = f"  \n*Context:* {f['context']}" if f.get("context") else ""
                st.markdown(f"- **{f['finding']}**{val}{ctx}")

        # Retrieved literature
        with st.expander("Retrieved Literature Chunks", expanded=False):
            for i, chunk in enumerate(result.literature, 1):
                score = chunk.get("relevance_score", 0)
                st.markdown(f"**{i}. [{chunk['title']}]({chunk['url']})**  \n"
                            f"*Chunk ID:* `{chunk['chunk_id']}` | "
                            f"*Relevance:* {score:.3f}")
                st.markdown(f"> {chunk['text'][:400]}...")
                st.divider()

        # Pipeline timing
        with st.expander("Pipeline Timing", expanded=False):
            for step, t in result.timings.items():
                st.markdown(f"- **{step}:** {t:.2f}s")

        # ── Hospital Recommendations ──────────────────────────
        st.markdown("""
        <div class="hospital-header">
            <h3>Nearby Hospital Recommendations</h3>
            <p>Based on your top diagnosis and current location</p>
        </div>
        """, unsafe_allow_html=True)

        # Try browser geolocation only once (not on every rerun)
        if st.session_state.user_lat is None and HAS_JS_EVAL:
            if "geo_attempted" not in st.session_state:
                st.session_state.geo_attempted = True
                js_code = """
                await new Promise((resolve, reject) => {
                    if (!navigator.geolocation) {
                        resolve(null);
                    } else {
                        navigator.geolocation.getCurrentPosition(
                            (pos) => resolve(
                                pos.coords.latitude + ',' + pos.coords.longitude
                            ),
                            (err) => resolve(null),
                            {timeout: 8000, enableHighAccuracy: false}
                        );
                    }
                })
                """
                location_str = streamlit_js_eval(
                    js_expressions=js_code, key="geo_location"
                )

                if location_str and "," in str(location_str):
                    parts = str(location_str).split(",")
                    try:
                        st.session_state.user_lat = float(parts[0])
                        st.session_state.user_lon = float(parts[1])
                        st.session_state.location_source = "gps"
                    except ValueError:
                        pass

        if st.session_state.user_lat is not None:
            source_labels = {
                "gps": "Browser GPS",
                "autocomplete": "Photon autocomplete",
                "address": "Manual address (fallback)",
            }
            source_label = source_labels.get(
                st.session_state.location_source, "Manual address"
            )
            st.caption(
                f"Location: {source_label} "
                f"({st.session_state.user_lat:.4f}, "
                f"{st.session_state.user_lon:.4f})"
            )

            # Cache hospital results in session state to avoid re-querying on reruns
            if ("hospital_rec" not in st.session_state
                    or st.session_state.get("hospital_rec_key") != (
                        result.differential, st.session_state.user_lat,
                        st.session_state.user_lon, hospital_radius)):
                with st.spinner("Searching for nearby hospitals..."):
                    rec = recommend_hospitals(
                        differential_text=result.differential,
                        lat=st.session_state.user_lat,
                        lon=st.session_state.user_lon,
                        radius_km=hospital_radius,
                    )
                st.session_state.hospital_rec = rec
                st.session_state.hospital_rec_key = (
                    result.differential, st.session_state.user_lat,
                    st.session_state.user_lon, hospital_radius)
            else:
                rec = st.session_state.hospital_rec

            # Specialty and urgency
            urgency_class = f"urgency-{rec['urgency'].lower()}"
            st.markdown(
                f"**Recommended specialty:** {rec['specialty']} "
                f"<span class='urgency-badge {urgency_class}'>"
                f"{rec['urgency']}</span>",
                unsafe_allow_html=True,
            )
            if rec["condition"]:
                st.caption(f"Based on top diagnosis: {rec['condition']}")

            if rec["error"]:
                st.warning(rec["error"])
            elif rec["hospitals"]:
                # Interactive map
                try:
                    import folium
                    from streamlit_folium import st_folium

                    m = folium.Map(
                        location=[
                            st.session_state.user_lat,
                            st.session_state.user_lon,
                        ],
                        zoom_start=12,
                    )

                    # User marker
                    folium.Marker(
                        [st.session_state.user_lat,
                         st.session_state.user_lon],
                        popup="Your Location",
                        icon=folium.Icon(
                            color="blue", icon="user", prefix="fa"
                        ),
                    ).add_to(m)

                    # Hospital markers
                    for h in rec["hospitals"][:15]:
                        if h["specialty_match"]:
                            color = "green"
                        elif h["emergency"]:
                            color = "red"
                        else:
                            color = "gray"

                        popup_parts = [f"<b>{h['name']}</b>"]
                        popup_parts.append(
                            f"Type: {h['type']} | "
                            f"{h['distance_km']} km"
                        )
                        if h["phone"]:
                            popup_parts.append(f"Phone: {h['phone']}")
                        if h["specialty"]:
                            popup_parts.append(
                                f"Specialty: {h['specialty']}"
                            )
                        popup_html = "<br>".join(popup_parts)

                        folium.Marker(
                            [h["lat"], h["lon"]],
                            popup=folium.Popup(
                                popup_html, max_width=250
                            ),
                            icon=folium.Icon(
                                color=color,
                                icon="plus-square",
                                prefix="fa",
                            ),
                        ).add_to(m)

                    # returned_objects=[] prevents map clicks from triggering reruns
                    st_folium(m, width=700, height=400, returned_objects=[])

                except ImportError:
                    # Fallback to st.map if folium not installed
                    import pandas as pd
                    map_data = pd.DataFrame([
                        {"lat": h["lat"], "lon": h["lon"]}
                        for h in rec["hospitals"][:15]
                    ])
                    st.map(map_data)

                # Hospital details list
                st.markdown("**Nearby Facilities:**")
                for i, h in enumerate(rec["hospitals"][:10], 1):
                    tags = []
                    if h["specialty_match"]:
                        tags.append("Specialty Match")
                    if h["emergency"]:
                        tags.append("Emergency Dept")
                    tag_str = (
                        " &mdash; " + ", ".join(f"**{t}**" for t in tags)
                        if tags else ""
                    )

                    details = (
                        f"**{i}. {h['name']}**{tag_str}  \n"
                        f"{h['distance_km']} km away | {h['type']}"
                    )
                    if h["address"]:
                        details += f"  \nAddress: {h['address']}"
                    if h["phone"]:
                        details += f"  \nPhone: {h['phone']}"
                    if h["website"]:
                        details += f"  \n[Website]({h['website']})"

                    st.markdown(details)
                    if i < min(len(rec["hospitals"]), 10):
                        st.divider()
        else:
            st.info(
                "Location not available. Enter your address in the "
                "sidebar under **Location Settings** to see nearby "
                "hospital recommendations."
            )

    elif run_btn:
        st.warning("Please enter clinical notes before clicking Analyze.")
    else:
        st.info(
            "Enter clinical notes on the left and click **Analyze** to generate "
            "a differential diagnosis with literature citations."
        )
