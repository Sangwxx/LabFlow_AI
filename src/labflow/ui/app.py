"""Streamlit 首页。"""

import streamlit as st

from labflow.config.settings import get_settings
from labflow.ui.home_content import HomeContent, build_home_content


def render_stage_cards(content: HomeContent) -> None:
    """渲染四阶段能力卡片。"""

    columns = st.columns(len(content.stage_cards))
    for column, card in zip(columns, content.stage_cards, strict=False):
        column.markdown(
            f"""
            <div class="stage-card">
                <div class="stage-step">{card.step}</div>
                <div class="stage-title">{card.title}</div>
                <div class="stage-body">{card.description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_scope(title: str, items: tuple[str, ...]) -> None:
    """渲染列表型信息块。"""

    st.markdown(f"### {title}")
    for item in items:
        st.markdown(f"- {item}")


def run() -> None:
    """运行首页界面。"""

    settings = get_settings()
    content = build_home_content(settings)

    st.set_page_config(
        page_title="LabFlow AI",
        page_icon="LF",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+SC:wght@400;500;600;700&display=swap');

            html, body, [class*="css"]  {
                font-family: "IBM Plex Sans SC", "Microsoft YaHei", sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(
                        circle at top right,
                        rgba(255, 176, 120, 0.18),
                        transparent 28%
                    ),
                    radial-gradient(
                        circle at left 20%,
                        rgba(26, 166, 154, 0.16),
                        transparent 24%
                    ),
                    linear-gradient(180deg, #f7f3eb 0%, #f2eee6 100%);
            }

            .hero {
                padding: 1.8rem 2rem;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(20, 37, 58, 0.96), rgba(42, 80, 73, 0.92));
                color: #fff7eb;
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 24px 60px rgba(18, 28, 42, 0.16);
                margin-bottom: 1rem;
            }

            .hero-kicker {
                font-size: 0.88rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #ffd7a8;
                margin-bottom: 0.75rem;
            }

            .hero-title {
                font-size: 2.9rem;
                font-weight: 700;
                line-height: 1.08;
                margin-bottom: 0.8rem;
            }

            .hero-subtitle {
                font-size: 1.05rem;
                line-height: 1.75;
                max-width: 58rem;
                color: rgba(255, 247, 235, 0.9);
            }

            .metric-card {
                border-radius: 20px;
                padding: 1rem 1.1rem;
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(20, 37, 58, 0.08);
                box-shadow: 0 10px 30px rgba(20, 37, 58, 0.06);
            }

            .metric-label {
                font-size: 0.85rem;
                color: #49606f;
                margin-bottom: 0.4rem;
            }

            .metric-value {
                font-size: 1.35rem;
                font-weight: 700;
                color: #14253a;
            }

            .stage-card {
                height: 100%;
                min-height: 220px;
                border-radius: 22px;
                padding: 1.15rem;
                background: rgba(255, 252, 246, 0.86);
                border: 1px solid rgba(20, 37, 58, 0.08);
                box-shadow: 0 10px 30px rgba(20, 37, 58, 0.06);
            }

            .stage-step {
                font-size: 0.82rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #0e7b71;
                margin-bottom: 0.8rem;
            }

            .stage-title {
                font-size: 1.22rem;
                font-weight: 700;
                color: #14253a;
                margin-bottom: 0.65rem;
            }

            .stage-body {
                font-size: 0.96rem;
                line-height: 1.75;
                color: #405261;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-kicker">HUST AI Competition / Stage 0</div>
            <div class="hero-title">{content.title}</div>
            <div class="hero-subtitle">{content.subtitle}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(3)
    metrics = (
        ("当前重点", "论文-代码对齐分析"),
        ("Git 输入", "本地路径导入"),
        ("PDF 范围", "仅文本型 PDF"),
    )
    for column, metric in zip(metric_columns, metrics, strict=False):
        label, value = metric
        column.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## 能力闭环")
    render_stage_cards(content)

    left_column, right_column = st.columns([1.1, 0.9])
    with left_column:
        render_scope("当前范围", content.current_scope)
    with right_column:
        render_scope("下一步实现", content.next_actions)

    st.info(
        "启动阶段已完成：目录结构、配置加载、敏感文件忽略与最小首页已就绪。"
        " 接下来可以进入 PDF 解析与本地仓库解析实现。"
    )
