"""Streamlit 页面样式。"""

from __future__ import annotations

import streamlit as st

APP_STYLES = """
<style>
    html, body, [class*="css"] { font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
    #MainMenu, footer, header[data-testid="stHeader"], div[data-testid="stDecoration"] { visibility: hidden; height: 0; }
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(227, 202, 168, 0.14), transparent 26%),
            radial-gradient(circle at left 22%, rgba(132, 158, 168, 0.1), transparent 24%),
            linear-gradient(180deg, #f7f3ec 0%, #f2ede5 100%);
    }
    .block-container { padding-top: 0.4rem; padding-bottom: 0rem; padding-left: 1rem; padding-right: 1rem; max-width: 98vw; }
    div[data-testid="column"] { width: 100% !important; }
    .landing-shell { max-width: 38rem; margin: 3.4rem auto 2rem auto; text-align: center; }
    .landing-kicker { font-size: 0.82rem; letter-spacing: 0.16em; text-transform: uppercase; color: #8b755d; margin-bottom: 0.95rem; }
    .landing-title { font-size: 2.6rem; line-height: 1.08; font-weight: 700; color: #182a3b; margin-bottom: 0.9rem; }
    .landing-body { max-width: 28rem; margin: 0 auto 0.8rem auto; font-size: 0.98rem; line-height: 1.7; color: #55697a; }
    .landing-status-line, .landing-action-hint { color: #7f6a52; font-size: 0.92rem; font-weight: 600; }
    .landing-action-hint { margin: 1.15rem 0 0.8rem 0; text-align: center; }
    .entry-card-head { margin-bottom: 0.8rem; text-align: left; }
    .entry-card-meta-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.6rem; }
    .entry-card-step { color: #7f6a52; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
    .entry-card-title { font-size: 1.32rem; line-height: 1.3; font-weight: 700; color: #1b3042; }
    .entry-card-state { flex: 0 0 auto; color: #8f7358; font-size: 0.82rem; font-weight: 700; }
    .entry-card-state-ready { color: #2e6f60; }
    .entry-card-desc { margin-top: 0.35rem; color: #5d6f7f; font-size: 0.93rem; line-height: 1.6; }
    .paper-preview-shell { margin-top: 0.95rem; padding: 0.95rem 1rem; border-radius: 16px; background: rgba(247, 243, 236, 0.96); border: 1px solid rgba(189, 174, 153, 0.32); }
    .paper-preview-source { color: #8b755d; font-size: 0.82rem; font-weight: 700; margin-bottom: 0.32rem; }
    .paper-preview-title { color: #1b3042; font-size: 1.02rem; font-weight: 700; line-height: 1.5; margin-bottom: 0.35rem; }
    .paper-preview-authors { color: #5b6d7d; font-size: 0.88rem; line-height: 1.6; margin-bottom: 0.65rem; }
    .paper-preview-abstract { color: #4d6072; font-size: 0.9rem; line-height: 1.7; margin-bottom: 0.7rem; }
    .paper-preview-meta { display: flex; flex-wrap: wrap; gap: 0.42rem; }
    .paper-preview-meta-chip { padding: 0.2rem 0.55rem; border-radius: 999px; background: rgba(32, 52, 74, 0.06); color: #4d6072; font-size: 0.8rem; line-height: 1.4; }
    .paper-preview-link { display: inline-flex; margin-top: 0.72rem; color: #1f4e7a; font-size: 0.84rem; font-weight: 600; text-decoration: none; }
    .paper-preview-link:hover { text-decoration: underline; }
    .repo-preview-shell { margin-top: 0.95rem; padding: 0.9rem 0.95rem; border-radius: 16px; background: rgba(247, 243, 236, 0.96); border: 1px solid rgba(189, 174, 153, 0.32); }
    .quick-guide-head { margin: 0.15rem 0 0.9rem 0; }
    .quick-guide-headline-text { color: #1b3042; font-size: 1.18rem; font-weight: 700; margin-bottom: 0.25rem; }
    .quick-guide-head-desc { color: #5d6f7f; font-size: 0.92rem; line-height: 1.65; }
    .quick-guide-shell { margin-top: 0.15rem; padding: 0.15rem 0; }
    .quick-guide-headline { color: #1b3042; font-size: 1.06rem; line-height: 1.6; font-weight: 700; margin-bottom: 0.75rem; }
    .quick-guide-stack { display: grid; gap: 0.7rem; }
    .quick-guide-item { padding: 0.88rem 0.95rem; border-radius: 14px; background: rgba(255, 255, 255, 0.72); border: 1px solid rgba(189, 174, 153, 0.24); }
    .quick-guide-label { color: #7f6a52; font-size: 0.8rem; font-weight: 700; margin-bottom: 0.38rem; }
    .quick-guide-body { color: #465a6d; font-size: 0.9rem; line-height: 1.72; }
    .guide-page-shell { margin: 0.45rem 0 0.15rem 0; text-align: left; }
    .guide-page-kicker { font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase; color: #8b755d; margin-bottom: 0.2rem; }
    .guide-page-title { font-size: 1.9rem; line-height: 1.12; font-weight: 700; color: #182a3b; margin-bottom: 0.25rem; }
    .guide-page-body { font-size: 0.95rem; line-height: 1.6; color: #55697a; }
    .repo-preview-head { margin-bottom: 0.75rem; }
    .repo-preview-title { color: #1b3042; font-size: 0.92rem; font-weight: 700; margin-bottom: 0.2rem; }
    .repo-preview-desc { color: #667889; font-size: 0.84rem; line-height: 1.6; }
    .repo-preview-group { padding-top: 0.7rem; border-top: 1px solid rgba(31, 43, 61, 0.08); }
    .repo-preview-group:first-of-type { padding-top: 0; border-top: 0; }
    .repo-preview-group-head { display: flex; align-items: center; justify-content: space-between; gap: 0.8rem; margin-bottom: 0.48rem; }
    .repo-preview-root { color: #1b3042; font-size: 0.9rem; font-weight: 700; }
    .repo-preview-count { color: #7b6b58; font-size: 0.8rem; }
    .repo-preview-children { display: flex; flex-wrap: wrap; gap: 0.4rem; }
    .repo-preview-child { padding: 0.2rem 0.55rem; border-radius: 999px; background: rgba(32, 52, 74, 0.06); color: #4d6072; font-size: 0.82rem; line-height: 1.4; }
    .repo-preview-child-muted { background: rgba(157, 137, 113, 0.12); color: #7f6a52; }
    div[data-testid="column"] [data-testid="stFileUploaderDropzone"] { border-radius: 18px; border: 1.5px dashed rgba(157, 137, 113, 0.3); background: #fcfaf6; }
    div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0.5rem 0.55rem 0.25rem 0.55rem; border-radius: 22px; border: 1px solid rgba(208, 193, 173, 0.72); background: rgba(255, 255, 255, 0.8); box-shadow: 0 14px 32px rgba(32, 48, 64, 0.06); }
    div[data-testid="column"] [data-testid="stTextInputRootElement"] input { min-height: 3.1rem; border-radius: 14px; border: 1px solid rgba(189, 174, 153, 0.5); background: #fcfaf6; }
    div[data-testid="column"] [data-testid="stFileUploader"], div[data-testid="column"] [data-testid="stTextInput"] { padding: 0.25rem 0 0.05rem 0; border-radius: 0; background: transparent; border: 0; box-shadow: none; }
    div[data-testid="column"] [data-testid="stFileUploader"] small, div[data-testid="column"] [data-testid="stTextInput"] small { color: #7f7264; }
    @media (max-width: 900px) { .landing-shell { margin-top: 2.2rem; } .landing-title { font-size: 2rem; } .guide-page-title { font-size: 1.65rem; } }
    div[data-testid="stButton"] > button { min-height: 3.05rem; border-radius: 14px; border: 0; background: linear-gradient(180deg, #20344a 0%, #162638 100%); color: #f7f2ea; font-weight: 700; box-shadow: 0 14px 24px rgba(22, 38, 56, 0.16); }
    div[data-testid="stButton"] > button:hover { background: linear-gradient(180deg, #263d56 0%, #1b2d42 100%); color: #fffaf2; }
    .workspace-title { font-size: 2.4rem; font-weight: 700; color: #1a2b3d; }
    .section-focus-bar { display: flex; align-items: center; gap: 0.75rem; min-height: 3.25rem; padding: 0.75rem 0.95rem; margin: 0 0 0.65rem 0; border-radius: 18px; background: rgba(239, 241, 246, 0.92); border: 1px solid rgba(34, 47, 62, 0.08); }
    .section-focus-page { flex: 0 0 auto; padding: 0.2rem 0.6rem; border-radius: 999px; background: rgba(31, 43, 61, 0.08); color: #1a2b3d; font-weight: 600; font-size: 0.92rem; }
    .section-focus-title { color: #2f4052; font-size: 1rem; font-weight: 500; line-height: 1.5; }
    .workspace-bar { display: flex; align-items: center; min-height: 3.35rem; padding: 0; }
    .code-panel-empty { display: flex; align-items: center; justify-content: center; min-height: 16rem; border-radius: 24px; background: rgba(252, 249, 243, 0.9); border: 1px solid rgba(189, 174, 153, 0.34); color: #53677a; text-align: center; font-size: 0.98rem; line-height: 1.9; padding: 2rem 1.6rem; }
    .source-overview-card { border-radius: 20px; padding: 1rem 1.05rem; background: rgba(245, 247, 250, 0.92); border: 1px solid rgba(34, 47, 62, 0.08); margin-bottom: 0.8rem; }
    .source-overview-meta { color: #59697a; font-size: 0.92rem; margin-bottom: 0.35rem; }
    .source-overview-title { color: #1a2b3d; font-size: 1rem; font-weight: 700; line-height: 1.5; margin-bottom: 0.45rem; }
    .source-overview-body { color: #4d6072; font-size: 0.95rem; line-height: 1.7; }
    .source-overview-shell { display: grid; gap: 0.9rem; margin-bottom: 0.85rem; }
    .source-meta-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.7rem; }
    .source-meta-card, .source-structure-shell, .source-guide-shell, .source-detail-block { border-radius: 18px; border: 1px solid rgba(31, 43, 61, 0.08); background: rgba(248, 250, 252, 0.96); }
    .source-meta-card { padding: 0.9rem 0.95rem; }
    .source-meta-label { color: #7a8a99; font-size: 0.82rem; margin-bottom: 0.3rem; }
    .source-meta-value { color: #1d3145; font-size: 0.98rem; font-weight: 700; line-height: 1.4; word-break: break-word; }
    .source-section-title, .source-detail-title { color: #1d3145; font-size: 0.96rem; font-weight: 700; margin-bottom: 0.55rem; }
    .source-structure-shell, .source-guide-shell, .source-detail-block { padding: 0.95rem 1rem; }
    .source-structure-item { color: #53677a; font-size: 0.92rem; line-height: 1.7; padding: 0.3rem 0; border-top: 1px solid rgba(31, 43, 61, 0.06); }
    .source-structure-item:first-of-type { border-top: 0; padding-top: 0; }
    .source-guide-shell { display: grid; gap: 0.55rem; }
    .source-guide-card { padding: 0.8rem 0.9rem; border-radius: 14px; background: rgba(244, 239, 231, 0.55); border: 1px solid rgba(157, 137, 113, 0.16); }
    .source-guide-meta { display: flex; flex-wrap: wrap; gap: 0.45rem 0.7rem; margin-bottom: 0.38rem; }
    .source-guide-symbol { color: #1a2b3d; font-size: 0.93rem; font-weight: 700; }
    .source-guide-range { color: #7a6a58; font-size: 0.84rem; }
    .source-guide-summary, .source-detail-body { color: #53677a; font-size: 0.92rem; line-height: 1.7; }
    .source-guide-reason { color: #6a7b8a; font-size: 0.86rem; line-height: 1.6; margin-top: 0.32rem; }
    @media (max-width: 900px) { .source-meta-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    .highlight-code-shell { border-radius: 22px; overflow: hidden; border: 1px solid rgba(20, 32, 46, 0.1); background: #f4efe7; }
    .highlight-code-scroll { overflow-x: auto; overflow-y: auto; max-height: 52rem; }
    .highlight-code-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    .highlight-code-line { width: 4.8rem; padding: 0.18rem 0.85rem; text-align: right; vertical-align: top; color: #8a755d; font-size: 0.84rem; border-right: 1px solid rgba(38, 55, 74, 0.08); user-select: none; white-space: nowrap; }
    .highlight-code-content { padding: 0.18rem 1rem; color: #1f2f42; font-size: 0.92rem; line-height: 1.6; white-space: pre-wrap; word-break: break-word; font-family: "Cascadia Code", "JetBrains Mono", "Consolas", monospace; }
    .highlight-code-row-highlight .highlight-code-line, .highlight-code-row-highlight .highlight-code-content { background: rgba(255, 223, 120, 0.2); }
    .trace-card { border-radius: 18px; padding: 0.9rem 1rem; background: rgba(248, 250, 252, 0.96); border: 1px solid rgba(31, 43, 61, 0.08); margin-bottom: 0.65rem; }
    .trace-card-title { color: #203349; font-size: 0.98rem; font-weight: 700; margin-bottom: 0.32rem; }
    .trace-card-body { color: #53677a; font-size: 0.92rem; line-height: 1.65; }
</style>
"""


def inject_styles() -> None:
    st.markdown(APP_STYLES, unsafe_allow_html=True)
