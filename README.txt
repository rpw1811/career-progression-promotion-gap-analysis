CAREER PROGRESSION STREAMLIT DASHBOARD

1. Place app.py and requirements.txt in your Career_Progression_Analysis folder.
2. Keep your existing data/ and outputs/ folders beside app.py.
3. Ensure this file exists:
   outputs/retention_analysis/employee_retention_opportunities.csv
4. Install:
   python -m pip install -r requirements.txt
5. Run:
   streamlit run app.py

Pages:
- Career Path Clustering
- Promotion Gap Monitor
- Retention Opportunity
- Managerial Insights

The app reads actual Phase 4/5 outputs. It does not fabricate missing data.
If the Phase 5 employee file is absent, it shows an instruction instead.

Important: The Retention Opportunity Index is a rule-based intervention
screening score, not a validated probability of employee resignation.
Verify dataset provenance before describing it as Palo Alto Networks data.
