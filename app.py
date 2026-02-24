"""
MGOC15 Lecture 7: AI Analytics Challenge — Streamlit App

HOW TO DEPLOY:
1. Create a GitHub repo and push this file (app.py) + requirements.txt
2. Go to https://share.streamlit.io
3. Connect your GitHub account, select the repo, and deploy
4. Share the URL with students on class day

HOW TO RUN LOCALLY:
  pip install streamlit pandas --break-system-packages
  streamlit run app.py
"""

import streamlit as st
import datetime
import json
import os
import csv
from pathlib import Path

# ============================================================
# CONFIGURATION — Edit these to match your generated dataset
# ============================================================

# Correct answers for Challenges 0–2
CORRECT_ANSWERS = {
    "c0": {
        "a": "Television",
    },
    "c1": {
        "a": "YouTube",
        "b": "Television",
    },
    "c2": {
        "a": "Television",
        "b": "Email Newsletter",
    },
}

# Correct answers for Challenge 3
# Pair 1 must be exact. Pairs 2–5 must all be in the correct set (any order).
C3_PAIR1 = ("Podcasts", "Working Parents")
C3_PAIRS_2_TO_5 = {
    ("YouTube", "Young Professionals"),
    ("YouTube", "Sports Enthusiasts"),
    ("Email Newsletter", "Retirees"),
    ("Email Newsletter", "Working Parents"),
}

CHANNELS = ["Podcasts", "Television", "YouTube", "Email Newsletter"]
DEMOGRAPHICS = [
    "Working Parents",
    "College Students",
    "Sports Enthusiasts",
    "Retirees",
    "Young Professionals",
]

# File to store responses (works locally and on Streamlit Cloud)
RESPONSES_FILE = "responses.csv"

# ============================================================
# INSIGHT TEXTS — shown at the top of the NEXT challenge
# ============================================================

INSIGHTS = {
    "c0": (
        "🔑 **Insight from Challenge 0 — Survivorship Bias:** "
        "Missing data almost always means the campaign failed. When you ignore those failures, "
        "you're only looking at winners — that's called **survivorship bias**. Television looked "
        "competitive until you counted its failures. **Use zero-filled revenue for the rest of your analysis.**"
    ),
    "c1": (
        "🔑 **Insight from Challenge 1 — Trends Matter:** "
        "YouTube has been climbing steadily. Television has been sliding. If you average across all history, "
        "you dilute YouTube's recent strength and mask Television's decline. "
        "**Recent data is a better predictor of next quarter than all-time averages.**"
    ),
    "c2": (
        "🔑 **Insight from Challenge 2 — Context Changes Everything:** "
        "Television collapses under heavy competition — its ad slots are auction-based, so intense competition "
        "inflates prices dramatically. Email Newsletter barely notices because it doesn't compete for the same "
        "attention space. Email Newsletter also performs especially well in Canadian markets. "
        "**With competitor intensity at 8/10 and a 65% Canadian footprint, these differences matter enormously.**"
    ),
    "c3": (
        "🔑 **Insight from Challenge 3 — Synthesis:** "
        "The pairs that perform best under NorthStar's specific conditions are very different from the pairs "
        "that look best on naive platform-wide averages. The analysis you just ran — combining survivorship "
        "correction, trend awareness, competition sensitivity, and regional focus — is exactly the kind of "
        "layered thinking that separates a good analyst from someone who just asks AI for the answer."
    ),
}


# ============================================================
# RESPONSE LOGGING
# ============================================================

def log_response(student_name, student_id, event_type, data):
    """Append a row to the responses CSV."""
    file_exists = Path(RESPONSES_FILE).exists()
    with open(RESPONSES_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "student_name", "student_id", "event_type", "data"])
        writer.writerow([
            datetime.datetime.now().isoformat(),
            student_name,
            student_id,
            event_type,
            json.dumps(data),
        ])


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_state():
    defaults = {
        "registered": False,
        "student_name": "",
        "student_id": "",
        "current_challenge": 0,  # 0, 1, 2, 3, or 4 (4 = allocation page)
        "max_challenge": 0,  # highest challenge reached (for back-navigation)
        "attempts": {"c0": 0, "c1": 0, "c2": 0, "c3": 0},
        "c3_submitted_pairs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MGOC15 AI Analytics Challenge",
    page_icon="📊",
    layout="centered",
)

init_state()


# ============================================================
# ADMIN PAGE — Secret download route
# ============================================================
# Access by adding ?admin=mgoc15 to the URL (change the password below)

ADMIN_PASSWORD = "mgoc15"  # ← Change this to any secret word you like

query_params = st.query_params
if query_params.get("admin") == ADMIN_PASSWORD:
    st.title("🔐 Admin — Download Responses")
    st.markdown("---")

    if Path(RESPONSES_FILE).exists():
        import io
        with open(RESPONSES_FILE, "r") as f:
            csv_data = f.read()
        num_lines = csv_data.count("\n") - 1  # minus header
        st.metric("Total logged events", num_lines)

        # Filter to just final allocations for a clean download
        reader = csv.DictReader(io.StringIO(csv_data))
        alloc_rows = [r for r in reader if r.get("event_type") == "allocation_final"]
        st.metric("Final allocation submissions", len(alloc_rows))

        st.markdown("### Download all responses (full log)")
        st.download_button(
            "📥 Download Full CSV",
            data=csv_data,
            file_name=f"mgoc15_responses_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        if alloc_rows:
            st.markdown("### Download final allocations only")
            alloc_output = io.StringIO()
            writer = csv.DictWriter(alloc_output, fieldnames=alloc_rows[0].keys())
            writer.writeheader()
            writer.writerows(alloc_rows)
            st.download_button(
                "📥 Download Allocations Only",
                data=alloc_output.getvalue(),
                file_name=f"mgoc15_allocations_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
    else:
        st.warning("No responses recorded yet.")

    st.stop()


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
    .insight-box {
        background-color: #e8f4e8;
        border-left: 5px solid #2e7d32;
        padding: 16px 20px;
        border-radius: 4px;
        margin-bottom: 24px;
    }
    .challenge-header {
        background-color: #f0f4ff;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #1a56db;
        margin-bottom: 20px;
    }
    .error-box {
        background-color: #fde8e8;
        border-left: 5px solid #c81e1e;
        padding: 12px 16px;
        border-radius: 4px;
    }
    .success-box {
        background-color: #e8f4e8;
        border-left: 5px solid #2e7d32;
        padding: 12px 16px;
        border-radius: 4px;
    }
    div[data-testid="stForm"] {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# REGISTRATION PAGE
# ============================================================

if not st.session_state.registered:
    st.title("📊 MGOC15: AI Analytics Challenge")
    st.markdown("---")
    st.markdown(
        "**Welcome to the AdPulse Digital Analytics Challenge!**\n\n"
        "You are an analyst at AdPulse Digital, a North American advertising platform. "
        "Your client — **NorthStar Retail** — has a **$100,000 budget** for Q1 2026 and needs your "
        "recommendation on how to allocate it across advertising channels and customer demographics.\n\n"
        "You have access to AdPulse's historical campaign data in your Jupyter notebook. "
        "Use AI to write code, but **you** must be the analyst — deciding what questions to ask, "
        "reading the results, and drawing conclusions.\n\n"
        "Work through each challenge in order. You cannot skip ahead."
    )

    st.markdown("---")
    st.subheader("Register")
    with st.form("registration"):
        name = st.text_input("Full Name")
        sid = st.text_input("Student ID")
        submitted = st.form_submit_button("Begin Challenge", type="primary")
        if submitted:
            if name.strip() and sid.strip():
                st.session_state.student_name = name.strip()
                st.session_state.student_id = sid.strip()
                st.session_state.registered = True
                log_response(name.strip(), sid.strip(), "registration", {})
                st.rerun()
            else:
                st.error("Please enter both your name and student ID.")
    st.stop()


# ============================================================
# SIDEBAR — Progress tracker
# ============================================================

with st.sidebar:
    st.markdown(f"**Student:** {st.session_state.student_name}")
    st.markdown(f"**ID:** {st.session_state.student_id}")
    st.markdown("---")
    st.markdown("### Progress")
    challenges = ["Challenge 0", "Challenge 1", "Challenge 2", "Challenge 3", "Allocation"]

    # Track which challenge the student has actually reached (highest unlocked)
    if "max_challenge" not in st.session_state:
        st.session_state.max_challenge = st.session_state.current_challenge
    st.session_state.max_challenge = max(
        st.session_state.max_challenge, st.session_state.current_challenge
    )

    for i, label in enumerate(challenges):
        if i < st.session_state.current_challenge:
            # Completed — clickable button to revisit
            if st.button(f"✅ {label}", key=f"nav_{i}"):
                st.session_state.current_challenge = i
                st.rerun()
        elif i == st.session_state.current_challenge:
            st.markdown(f"➡️ **{label}** ← you are here")
        elif i <= st.session_state.max_challenge:
            # Previously reached but navigated back — allow forward to here
            if st.button(f"✅ {label}", key=f"nav_{i}"):
                st.session_state.current_challenge = i
                st.rerun()
        else:
            st.markdown(f"🔒 {label}")

    st.markdown("---")
    st.markdown("### Conditions (from the board)")
    st.markdown(
        "- Budget: **$100,000**\n"
        "- Competitor intensity: **8/10**\n"
        "- Regional mix: **65% Canada, 35% US**\n"
        "- Minimum per pair: **$3,000**"
    )


# ============================================================
# HELPER: Render a challenge
# ============================================================

def render_challenge(
    challenge_key,
    title,
    description,
    questions,
    correct_answers_dict,
    prev_insight_key=None,
):
    """
    Render a challenge page with all questions on one screen.

    Parameters:
        challenge_key: e.g., "c0"
        title: display title
        description: the challenge prompt
        questions: list of dicts with keys: "key", "label", "options"
        correct_answers_dict: dict mapping question key to correct answer string
        prev_insight_key: if set, show the insight from this challenge at the top
    """

    st.title(f"📊 {title}")

    # Show insight from previous challenge
    if prev_insight_key and prev_insight_key in INSIGHTS:
        st.markdown(INSIGHTS[prev_insight_key])
        st.markdown("---")

    # Challenge description
    st.markdown(description)

    st.markdown("")

    # Form with all questions
    with st.form(f"form_{challenge_key}"):
        answers = {}
        for q in questions:
            answers[q["key"]] = st.radio(
                q["label"],
                options=["— Select —"] + q["options"],
                index=0,
                key=f"{challenge_key}_{q['key']}",
            )

        submitted = st.form_submit_button("Submit", type="primary")

    if submitted:
        # Check all selected
        unselected = [q["label"] for q in questions if answers[q["key"]] == "— Select —"]
        if unselected:
            st.error("Please answer all questions before submitting.")
            return

        # Check correctness
        all_correct = all(
            answers[q["key"]] == correct_answers_dict[q["key"]]
            for q in questions
        )

        st.session_state.attempts[challenge_key] = st.session_state.attempts.get(challenge_key, 0) + 1
        attempt_num = st.session_state.attempts[challenge_key]

        log_response(
            st.session_state.student_name,
            st.session_state.student_id,
            f"{challenge_key}_attempt",
            {
                "attempt": attempt_num,
                "answers": {q["key"]: answers[q["key"]] for q in questions},
                "correct": all_correct,
            },
        )

        if all_correct:
            st.success("✅ Correct! Advancing to the next challenge...")
            st.balloons()
            st.session_state.current_challenge += 1
            st.rerun()
        else:
            # Build feedback
            wrong_qs = [
                q["label"]
                for q in questions
                if answers[q["key"]] != correct_answers_dict[q["key"]]
            ]
            st.error(
                f"❌ Not quite — {len(wrong_qs)} of {len(questions)} answer(s) incorrect. "
                f"(Attempt #{attempt_num})"
            )

            # Show hints after 3 failures
            if attempt_num >= 3:
                with st.expander("💡 Hint (click to expand)"):
                    hints = {
                        "c0": (
                            "Try asking AI:\n\n"
                            "```\n"
                            "# Compare ROI before and after zero-fill\n"
                            "df['roi_original'] = df['revenue'] / df['cost']\n"
                            "df['revenue_filled'] = df['revenue'].fillna(0)\n"
                            "df['roi_filled'] = df['revenue_filled'] / df['cost']\n\n"
                            "# Compare averages by channel\n"
                            "before = df.groupby('channel')['roi_original'].mean()\n"
                            "after = df.groupby('channel')['roi_filled'].mean()\n"
                            "print(before - after)  # Which channel dropped the most?\n"
                            "```"
                        ),
                        "c1": (
                            "Try asking AI to create a line chart of average ROI by channel for each quarter. "
                            "Look at which line is climbing and which is falling.\n\n"
                            "```\n"
                            "df['roi'] = df['revenue_filled'] / df['cost']\n"
                            "df.groupby(['quarter','channel'])['roi'].mean().unstack('channel').plot()\n"
                            "```"
                        ),
                        "c2": (
                            "For Question A, compare each channel's ROI under low competition (≤ 3) vs. high (≥ 7). "
                            "For Question B, filter to Canadian regions (Ontario, British Columbia, Quebec, Alberta) "
                            "and compare each channel's ROI there vs. its overall average.\n\n"
                            "```\n"
                            "low = df[df['competitor_intensity'] <= 3]\n"
                            "high = df[df['competitor_intensity'] >= 7]\n"
                            "# Compare: high.groupby('channel')['roi'].mean() vs low\n"
                            "```"
                        ),
                        "c3": (
                            "Think about what you learned in Challenges 0–2. What filters would create a subset "
                            "of the data that best matches NorthStar's conditions next quarter? Consider:\n"
                            "- What did you learn about missing data?\n"
                            "- What did you learn about trends over time?\n"
                            "- What did you learn about competition and geography?"
                        ),
                    }
                    st.markdown(hints.get(challenge_key, "Review your earlier insights."))


# ============================================================
# CHALLENGE 0
# ============================================================

if st.session_state.current_challenge == 0:
    render_challenge(
        challenge_key="c0",
        title='Challenge 0 — "What\'s Missing?"',
        description=(
            "Before you can trust this data, you need to understand what's hiding in it.\n\n"
            "The `revenue` column has missing values — campaigns where revenue was never recorded. "
            "Missing data is rarely random. Your first job: figure out how bad the problem is "
            "and what it does to your numbers.\n\n"
            "Compute the average ROI (revenue ÷ cost) for each channel two ways: first ignoring "
            "missing values entirely, then filling all missing revenue with zero and recomputing. "
            "Compare the results."
        ),
        questions=[
            {
                "key": "a",
                "label": (
                    "Which channel's average ROI drops the most when you fill missing revenue "
                    "with zero instead of ignoring it?"
                ),
                "options": CHANNELS,
            },
        ],
        correct_answers_dict=CORRECT_ANSWERS["c0"],
    )
    st.stop()


# ============================================================
# CHALLENGE 1
# ============================================================

if st.session_state.current_challenge == 1:
    render_challenge(
        challenge_key="c1",
        title='Challenge 1 — "Spot the Shift"',
        description=(
            "Averages across two years of data treat every quarter equally. But channels don't stand still — "
            "some are growing, some are fading.\n\n"
            "If you're planning for *next* quarter, you need to know which direction things are heading.\n\n"
            "Investigate how each channel's ROI has changed over the 12 quarters in the dataset."
        ),
        questions=[
            {
                "key": "a",
                "label": "Question A: Which channel has shown the strongest improvement in ROI over the past 12 quarters?",
                "options": CHANNELS,
            },
            {
                "key": "b",
                "label": "Question B: Which channel has declined the most over the same period?",
                "options": CHANNELS,
            },
        ],
        correct_answers_dict=CORRECT_ANSWERS["c1"],
        prev_insight_key="c0",
    )
    st.stop()


# ============================================================
# CHALLENGE 2
# ============================================================

if st.session_state.current_challenge == 2:
    render_challenge(
        challenge_key="c2",
        title='Challenge 2 — "Know Your Battlefield"',
        description=(
            "NorthStar's upcoming quarter has specific conditions: high competitor intensity and a "
            "footprint heavily weighted toward Canada.\n\n"
            "Not all channels respond the same way to competition, and not all channels perform equally "
            "across regions.\n\n"
            "Investigate how competitor intensity and geography affect each channel's performance."
        ),
        questions=[
            {
                "key": "a",
                "label": (
                    "Question A: Under high competitor intensity (7 or above), which channel's ROI "
                    "suffers the most compared to its performance under low intensity (3 or below)?"
                ),
                "options": CHANNELS,
            },
            {
                "key": "b",
                "label": (
                    "Question B: Looking only at Canadian regions, which channel performs strongest "
                    "relative to its overall platform-wide average?"
                ),
                "options": CHANNELS,
            },
        ],
        correct_answers_dict=CORRECT_ANSWERS["c2"],
        prev_insight_key="c1",
    )
    st.stop()


# ============================================================
# CHALLENGE 3
# ============================================================

if st.session_state.current_challenge == 3:
    st.title('📊 Challenge 3 — "Where the Money Is"')

    # Show insight from Challenge 2
    st.markdown(INSIGHTS["c2"])
    st.markdown("---")

    st.markdown(
        "You now know which channels are trending up or down, which ones fail silently, "
        "which ones crumble under competition, and where the regional strengths lie.\n\n"
        "NorthStar needs you to identify the specific **channel–demographic combinations** that will "
        "deliver the best return under their conditions next quarter.\n\n"
        "Using what you've learned in Challenges 0–2, identify the **top 5 channel–demographic pairs by ROI** "
        "for NorthStar's upcoming quarter.\n\n"
        "*The challenge doesn't tell you what filters to apply. That's your job as the analyst. "
        "Think about what you've learned and decide which historical data best predicts NorthStar's future.*"
    )

    st.markdown("")

    with st.form("form_c3"):
        pairs = []
        for i in range(1, 6):
            label = "Pair 1 (Highest ROI)" if i == 1 else f"Pair {i}"
            col1, col2 = st.columns(2)
            with col1:
                ch = st.selectbox(
                    f"{label} — Channel",
                    options=["— Select —"] + CHANNELS,
                    key=f"c3_p{i}_ch",
                )
            with col2:
                dem = st.selectbox(
                    f"{label} — Demographic",
                    options=["— Select —"] + DEMOGRAPHICS,
                    key=f"c3_p{i}_dem",
                )
            pairs.append((ch, dem))

        submitted = st.form_submit_button("Submit", type="primary")

    if submitted:
        # Check all selected
        if any(ch == "— Select —" or dem == "— Select —" for ch, dem in pairs):
            st.error("Please select both a channel and demographic for all 5 pairs.")
        else:
            st.session_state.attempts["c3"] = st.session_state.attempts.get("c3", 0) + 1
            attempt_num = st.session_state.attempts["c3"]

            # Validate
            pair1_correct = (pairs[0][0] == C3_PAIR1[0] and pairs[0][1] == C3_PAIR1[1])

            submitted_pairs_2_to_5 = set()
            for ch, dem in pairs[1:]:
                submitted_pairs_2_to_5.add((ch, dem))

            pairs_2_to_5_correct = submitted_pairs_2_to_5 == C3_PAIRS_2_TO_5

            all_correct = pair1_correct and pairs_2_to_5_correct

            log_response(
                st.session_state.student_name,
                st.session_state.student_id,
                "c3_attempt",
                {
                    "attempt": attempt_num,
                    "pairs": [(ch, dem) for ch, dem in pairs],
                    "correct": all_correct,
                },
            )

            if all_correct:
                st.success("✅ All pairs correct! Advancing to the allocation phase...")
                st.balloons()
                st.session_state.current_challenge = 4
                st.rerun()
            else:
                # Specific feedback
                errors = []
                if not pair1_correct:
                    errors.append("Pair 1 (highest ROI) is incorrect.")
                if not pairs_2_to_5_correct:
                    errors.append("One or more of Pairs 2–5 are incorrect.")
                st.error(
                    f"❌ Not quite. (Attempt #{attempt_num})\n\n" + "\n".join(f"• {e}" for e in errors)
                )

                if attempt_num >= 3:
                    with st.expander("💡 Hint (click to expand)"):
                        st.markdown(
                            "Think about what you learned in Challenges 0–2. What filters would create a subset "
                            "of the data that best matches NorthStar's conditions next quarter? Consider:\n\n"
                            "- What did you learn about missing data? How should you handle it?\n"
                            "- What did you learn about trends over time? Should you use all 12 quarters equally?\n"
                            "- What did you learn about competition and geography? How do you match NorthStar's conditions?\n\n"
                            "Apply those lessons, then group by channel + demographic and rank by average ROI."
                        )
    st.stop()


# ============================================================
# ALLOCATION PAGE
# ============================================================

if st.session_state.current_challenge == 4:
    st.title("💰 Budget Allocation")

    # Show insight from Challenge 3
    st.markdown(INSIGHTS["c3"])
    st.markdown("---")

    st.markdown(
        "It's time to allocate NorthStar's **$100,000** budget across the 4×5 grid of "
        "channel–demographic pairs.\n\n"
        "Enter dollar amounts in each cell. Remember:\n"
        "- Total must equal exactly **$100,000**\n"
        "- Any pair with less than **$3,000** allocated generates **zero** revenue\n"
        "- Concentrating too much in one pair means diminishing returns\n"
    )

    st.markdown("---")

    # Build the allocation grid
    with st.form("allocation_form"):
        st.markdown("**Enter your allocation (in dollars):**")

        # Header row
        header_cols = st.columns([1.5] + [1] * len(DEMOGRAPHICS))
        with header_cols[0]:
            st.markdown("**Channel**")
        for j, dem in enumerate(DEMOGRAPHICS):
            with header_cols[j + 1]:
                # Shorten labels for display
                short = dem.replace("Working Parents", "Work. Par.")\
                           .replace("College Students", "Coll. Stu.")\
                           .replace("Sports Enthusiasts", "Sports")\
                           .replace("Young Professionals", "Young Pro.")
                st.markdown(f"**{short}**")

        # Data rows
        allocations = {}
        for ch in CHANNELS:
            cols = st.columns([1.5] + [1] * len(DEMOGRAPHICS))
            with cols[0]:
                st.markdown(f"**{ch}**")
            for j, dem in enumerate(DEMOGRAPHICS):
                with cols[j + 1]:
                    key = f"alloc_{ch}_{dem}"
                    allocations[(ch, dem)] = st.number_input(
                        f"{ch}-{dem}",
                        min_value=0,
                        max_value=100000,
                        value=0,
                        step=1000,
                        key=key,
                        label_visibility="collapsed",
                    )

        st.markdown("---")

        # Running total
        total = sum(allocations.values())
        remaining = 100000 - total

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            color = "green" if total == 100000 else ("red" if total > 100000 else "orange")
            st.markdown(f"**Total allocated:** :{color}[**${total:,}**] / $100,000")
        with col_t2:
            if remaining > 0:
                st.markdown(f"**Remaining:** ${remaining:,}")
            elif remaining < 0:
                st.markdown(f"**Over budget by:** :red[${-remaining:,}]")
            else:
                st.markdown("✅ **Budget fully allocated**")

        # Warnings for sub-$3K nonzero allocations
        warnings = []
        for (ch, dem), val in allocations.items():
            if 0 < val < 3000:
                warnings.append(f"⚠️ {ch} × {dem}: ${val:,} is below $3,000 minimum — will generate $0 revenue")
        if warnings:
            for w in warnings:
                st.warning(w)

        submitted = st.form_submit_button("🚀 Submit Final Allocation", type="primary")

    if submitted:
        if total != 100000:
            st.error(f"Your allocation totals ${total:,}. It must equal exactly $100,000.")
        else:
            # Save
            alloc_data = {f"{ch}|{dem}": val for (ch, dem), val in allocations.items()}
            alloc_data["total"] = total
            alloc_data["num_pairs"] = sum(1 for v in allocations.values() if v >= 3000)
            alloc_data["sub_3k_pairs"] = sum(1 for v in allocations.values() if 0 < v < 3000)

            log_response(
                st.session_state.student_name,
                st.session_state.student_id,
                "allocation_final",
                alloc_data,
            )

            st.success(
                "✅ **Allocation submitted!** Your response has been recorded.\n\n"
                f"You allocated across **{alloc_data['num_pairs']}** pairs."
            )
            if alloc_data["sub_3k_pairs"] > 0:
                st.warning(
                    f"⚠️ {alloc_data['sub_3k_pairs']} pair(s) are below $3,000 and will generate $0 revenue."
                )
            st.balloons()

    # "Have You Considered?" panel
    with st.expander("🤔 Have You Considered? (optional refinement prompts)"):
        st.markdown(
            "Before you finalize, some questions that might lead to a better allocation:\n\n"
            "- Have you checked whether some channels scale better than others? "
            "What does the `budget_tier` column tell you about ROI at different spending levels?\n\n"
            "- Your client's footprint is 65% Canadian next quarter. Have you looked at how channels "
            "perform specifically in Canadian regions (Ontario, BC, Quebec, Alberta)?\n\n"
            "- Some channels are more affected by competitor intensity than others. "
            "Did your analysis account for this beyond just filtering to intensity ≥ 6?\n\n"
            "- Have you looked at which `creative_type` drives the best performance for your top pairs?\n\n"
            "- Are you sure about the pairs you chose NOT to invest in? "
            "Is there a pair outside your top 5 that might still be worth $3K–$5K?"
        )
