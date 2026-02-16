"""Minimal Streamlit hello dashboard."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """Render a minimal dashboard smoke test."""
    st.set_page_config(page_title="AH Premium Lab - Hello", page_icon="📈", layout="centered")
    st.title("AH Premium Lab")
    st.subheader("Hello Dashboard")
    st.write("Streamlit app is running successfully.")


if __name__ == "__main__":
    main()
