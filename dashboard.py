import streamlit as st
import streamlit.components.v1 as components
import os
import re

st.set_page_config(layout="wide")
st.title("🛡️ Project Drift Sentinel")

if os.path.exists("drift_report.md"):
    with open("drift_report.md", "r") as f:
        content = f.read()

    # Extract Expected Outputs section
    outputs_match = re.search(r'## 📊 Expected Outputs(.*?)(?=\n##|\Z)', content, re.DOTALL)

    # Extract Mermaid diagram
    mermaid_match = re.search(r'```mermaid\n(.*?)\n```', content, re.DOTALL)

    # Show Expected Outputs prominently at the top
    if outputs_match:
        st.markdown("## 🎯 What You'll Get After This Update")
        outputs_section = outputs_match.group(1)

        # Create a nice bordered container for outputs
        st.markdown("""
        <style>
        .output-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin: 20px 0;
        }
        .output-content {
            background: white;
            padding: 20px;
            border-radius: 8px;
            color: #2d3748;
            margin-top: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="output-box"><h3 style="margin:0; color:white;">📦 Output Preview</h3><div class="output-content">', unsafe_allow_html=True)
        st.markdown(outputs_section)
        st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    if mermaid_match:
        mermaid_code = mermaid_match.group(1)

        # Render Mermaid diagram using mermaid.js CDN
        mermaid_html = f"""
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>
        </head>
        <body>
            <div class="mermaid">
                {mermaid_code}
            </div>
        </body>
        </html>
        """

        st.subheader("📊 Impact Blast Radius Diagram")
        st.caption("Visual map of code changes and dependencies")
        components.html(mermaid_html, height=600, scrolling=True)

        # Show remaining content (analysis details)
        text_content = content
        # Remove mermaid block
        text_content = re.sub(r'```mermaid\n.*?\n```', '', text_content, flags=re.DOTALL)
        # Remove expected outputs section (already shown above)
        text_content = re.sub(r'## 📊 Expected Outputs.*?(?=\n##|\Z)', '', text_content, flags=re.DOTALL)

        if text_content.strip():
            st.markdown("---")
            st.subheader("📝 Detailed Analysis")
            st.markdown(text_content)
    else:
        # No mermaid diagram found, just show content
        st.markdown(content)

    st.markdown("---")

    # Check approval status
    if os.path.exists("approved.signal"):
        st.success("✅ **APPROVED** - Claude is authorized to implement")
    elif os.path.exists("rejected.signal"):
        st.error("❌ **REJECTED** - Plan was rejected")
    else:
        st.info("⏳ **Awaiting Decision** - Tell Claude to 'approve' or 'reject' in chat")
        st.markdown("""
        ### How to proceed:
        - Type **"approve"** in chat → Claude creates `approved.signal` and implements
        - Type **"reject"** in chat → Claude creates `rejected.signal` and discards plan
        """)
else:
    st.info("⏳ Waiting for Claude to propose a plan...")
