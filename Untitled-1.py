"""Old main file name, kept so a Streamlit Cloud app still set to 'Untitled-1.py' keeps working.

The dashboard lives in dashboard.py. This runs it exactly as `streamlit run dashboard.py` would.
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.py'), run_name='__main__')
