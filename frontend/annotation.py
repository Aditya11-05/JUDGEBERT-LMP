"""Annotation Studio UI for JUDGEBERT-LMP."""

from __future__ import annotations

import streamlit as st

from app.data import annotation_store as store


def _render_pair(pair: dict) -> None:
    st.caption(
        f"Pair: {pair['pair_id']}  |  "
        f"Document: {pair.get('source_document_id') or 'Unresolved'}"
    )

    if pair.get("source_licence") == "UNVERIFIED":
        st.warning(
            "Source licence is unverified. This pair can be annotated for "
            "research review, but it is not eligible for model training."
        )

    left, right = st.columns(2)
    left.markdown("**Original legal text**")
    left.text_area(
        "Original text",
        value=pair["original_text"],
        height=220,
        disabled=True,
        key=f"original_{pair['pair_id']}",
        label_visibility="collapsed",
    )

    right.markdown("**Simplified legal text**")
    right.text_area(
        "Simplified text",
        value=pair["simplified_text"],
        height=220,
        disabled=True,
        key=f"simplified_{pair['pair_id']}",
        label_visibility="collapsed",
    )


def _annotate() -> None:
    st.markdown("#### Independent annotation")
    st.caption(
        "Submit your own judgment independently. Do not discuss ratings "
        "with other annotators before adjudication."
    )

    reviewer_id = st.text_input(
        "Annotator ID",
        key="annotation_reviewer_id",
        max_chars=80,
        placeholder="e.g. REVIEWER-A",
    ).strip()

    qualification = st.text_input(
        "Your qualifications",
        key="annotation_qualification",
        max_chars=250,
        placeholder="e.g. CS undergraduate, no legal training",
    ).strip()

    if not reviewer_id or not qualification:
        st.info("Enter your annotator ID and qualifications to load your next pair.")
        return

    try:
        pair = store.get_next_pair(reviewer_id)
    except Exception as exc:
        st.error(f"Could not load the annotation queue: {exc}")
        return

    if pair is None:
        st.success(
            "You have no remaining unannotated pairs in the current queue."
        )
        return

    _render_pair(pair)

    st.caption(
        "Evaluate the complete simplified passage against the complete original passage. "
        "Consider all legally relevant duties, rights, conditions, exceptions, definitions, "
        "parties, dates, amounts and scope. A higher score means better preservation."
    )

    with st.form(f"annotation_form_{pair['pair_id']}"):
        rating = st.selectbox(
            "Passage-level Legal Meaning Preservation rating",
            options=list(range(1, 11)),
            index=None,
            placeholder="Select a rating from 1 to 10",
            help=(
                "1 = fundamentally meaning-changing; "
                "9–10 = legally relevant meaning preserved throughout the passage."
            ),
        )

        confidence = st.selectbox(
            "Confidence in your rating",
            options=list(range(1, 6)),
            index=None,
            placeholder="Select confidence from 1 to 5",
        )

        categories = st.multiselect(
            "Observed error categories (leave empty if none)",
            options=sorted(store.ERROR_CATEGORIES),
            help=(
                "Select only changes you actually observe. "
                "Do not select a category merely because a rule-based "
                "system flagged it."
            ),
        )

        notes = st.text_area(
            "Annotation notes",
            max_chars=2000,
            placeholder="Briefly explain the legally relevant change, if any.",
        )

        submitted = st.form_submit_button(
            "Submit independent annotation",
            type="primary",
        )

    if submitted:
        if rating is None or confidence is None:
            st.error("Select both a passage-level rating and confidence before submitting.")
            return

        try:
            store.submit_annotation(
                pair_id=pair["pair_id"],
                annotator_id=reviewer_id,
                qualification=qualification,
                rating=rating,
                confidence=confidence,
                error_categories=categories,
                annotation_notes=notes,
            )
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not save annotation: {exc}")
        else:
            st.success(f"Annotation saved for {pair['pair_id']}.")
            st.rerun()


def _adjudicate() -> None:
    st.markdown("#### Adjudication")
    st.caption(
        "This section is for the designated adjudicator. "
        "Reviewer ratings and notes are visible here, not in the independent "
        "annotation section."
    )

    adjudicator_id = st.text_input(
        "Adjudicator ID",
        key="adjudicator_id",
        max_chars=80,
        placeholder="e.g. ADJUDICATOR-1",
    ).strip()

    if not adjudicator_id:
        st.info("Enter the adjudicator ID to view pending cases.")
        return

    try:
        pending = store.get_pairs_pending_adjudication()
    except Exception as exc:
        st.error(f"Could not load adjudication queue: {exc}")
        return

    if not pending:
        st.info("No pairs are currently awaiting adjudication.")
        return

    labels = {
        item["pair_id"]: (
            f"{item['pair_id']} — "
            f"{item['reviewer_count']} reviewers — "
            f"ratings {item['minimum_rating']}–{item['maximum_rating']}"
        )
        for item in pending
    }

    selected_id = st.selectbox(
        "Pair awaiting adjudication",
        options=list(labels),
        format_func=lambda pair_id: labels[pair_id],
        key="adjudication_pair",
    )

    pair = store.get_pair(selected_id)
    if pair is None:
        st.error("Selected pair could not be found.")
        return

    _render_pair(pair)

    try:
        reviews = store.get_adjudication_inputs(selected_id)
    except Exception as exc:
        st.error(f"Could not load reviewer submissions: {exc}")
        return

    st.markdown("**Independent reviewer submissions**")

    for index, review in enumerate(reviews, start=1):
        with st.expander(
            f"Reviewer {index}: {review['annotator_id']} — "
            f"rating {review['rating']}/10",
            expanded=True,
        ):
            st.write(f"Confidence: {review['confidence']}/5")
            st.write(
                "Error categories: "
                + (review["error_categories"] or "[]")
            )
            st.write(
                "Notes: "
                + (review["annotation_notes"] or "No notes supplied.")
            )
            st.caption(f"Submitted: {review['submitted_at']}")

    ratings = [int(review["rating"]) for review in reviews]
    spread = max(ratings) - min(ratings)

    st.metric("Rating disagreement spread", spread)
    if spread > 2:
        st.warning(
            "The reviewers differ by more than two points. "
            "Record how this disagreement was resolved."
        )

    with st.form(f"adjudication_form_{selected_id}"):
        final_rating = st.slider(
            "Final adjudicated LMP rating",
            min_value=1,
            max_value=10,
            value=round(sum(ratings) / len(ratings)),
            step=1,
        )

        resolution = st.text_area(
            "Adjudication resolution notes (required)",
            max_chars=3000,
            placeholder=(
                "Explain the final decision and how reviewer disagreement "
                "was resolved, where applicable."
            ),
        )

        submitted = st.form_submit_button(
            "Record adjudication",
            type="primary",
        )

    if submitted:
        if not resolution.strip():
            st.error("Resolution notes are required.")
            return

        try:
            store.record_adjudication(
                pair_id=selected_id,
                adjudicator_id=adjudicator_id,
                adjudicated_rating=final_rating,
                resolution_notes=resolution,
            )
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not save adjudication: {exc}")
        else:
            st.success(f"Adjudication saved for {selected_id}.")
            st.rerun()


def render_annotation_studio() -> None:
    """Render the Annotation Studio without changing global application styling."""
    st.markdown(
        "Collect independent human judgments of legal meaning preservation. "
        "Annotations are stored locally and separately from the source CSV."
    )

    try:
        summary = store.get_annotation_summary()
    except Exception as exc:
        st.error(f"Annotation database is unavailable: {exc}")
        return

    metrics = st.columns(4)
    metrics[0].metric("Candidate pairs", summary["candidate_pairs"])
    metrics[1].metric("Annotations", summary["annotations"])
    metrics[2].metric("Annotators", summary["annotators"])
    metrics[3].metric("Adjudications", summary["adjudications"])

    st.caption(
        "Local SQLite storage. Reviewer IDs are self-declared and are not "
        "authenticated accounts. Use this application on the designated "
        "computer and do not share reviewer credentials."
    )

    section = st.radio(
        "Annotation workflow",
        ["Annotate", "Adjudicate"],
        horizontal=True,
        key="annotation_workflow_section",
    )

    if section == "Annotate":
        _annotate()
    else:
        _adjudicate()
