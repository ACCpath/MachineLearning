import pathlib
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts
from executionmodelrisk import get_expected_loss

path_dir = pathlib.Path(__name__).parent.resolve()
# Page settings
st.set_page_config(page_title='Risk Score Analyzer', layout='wide')

# Side bar
with st.sidebar:
    full_path = path_dir / 'risk_score.jpg' 
    full_path = full_path.resolve()         
    st.image(str(full_path))
    #Input data
    principal = st.number_input('**Requested amount**', 500, 50_000)
    purpose_list = [
            'debt_consolidation', 'credit_card', 'home_improvement',
            'other', 'major_purchase', 'medical', 'small_business',
            'car', 'vacation'
            ]
    purpose =  st.selectbox('**Purpose of the loan**', purpose_list)
    dues = st.radio('**Dues number**', ['36 months', '60 months'])
    income = st.slider('**Annual income**', min_value=20000, max_value=300_000)

# Static data
income_verified = 'Verified'
employment_antiquity =  '10+ years'
rating = 'B'
dti = 28
num_credit_lines = 3
porc_uso_revolving = 50
type_interest = 7.26
fee_amount = 500
derogatory_number = 0
housing = 'MORTGAGE'

# Main
st.title('RISK SCORE ANALYZER')

# Generate record
record = pd.DataFrame({
    'ingresos_verificados': income_verified,
    'vivienda': housing,
    'finalidad': purpose,
    'num_cuotas': dues,
    'antigüedad_empleo': employment_antiquity,
    'rating': rating,
    'ingresos': income,
    'dti': dti,
    'num_lineas_credito': num_credit_lines,
    'porc_uso_revolving': porc_uso_revolving,
    'principal': principal,
    'tipo_interes': type_interest,
    'imp_cuota': fee_amount,
    'num_derogatorios': derogatory_number
    }, index=[0])

# Calculate risk
if st.sidebar.button('Calculate risk'):
    EL = get_expected_loss(record)
    #Calculate KPI
    kpi_pd = int(EL['pd'].iloc[0] * 100)
    kpi_ead = int(EL['ead'].iloc[0] * 100)
    kpi_lgd = int(EL['lgd'].iloc[0] * 100)
    kpi_el = int(EL['principal'].iloc[0] * EL['pd'].iloc[0] * EL['ead'].iloc[0] *
              EL['lgd'].iloc[0])

    # Speedmeters
    # source:https://towardsdatascience.com/5-streamlit-components-to-build-better-applications-71e0195c82d4

    #Speedmeter for pd
    pd_options = {
            "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
            "series": [{
                "name": "PD",
                "type": "gauge",
                "axisLine": {"lineStyle": {"width": 10,},},
                "progress": {"show": "true", "width": 10},
                "detail": {"valueAnimation": "true", "formatter": "{value}",
                           "textStyle": {"color": "white"},},
                "data": [{"value": kpi_pd, "name": "PD"}],
                }],
            }

    #Speedmeter for ead
    ead_options = {
            "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
            "series": [{
                "name": "EAD",
                "type": "gauge",
                "axisLine": {"lineStyle": {"width": 10,},},
                "progress": {"show": "true", "width": 10},
                "detail": {"valueAnimation": "true", "formatter": "{value}",
                           "textStyle": {"color": "white"},},
                "data": [{"value": kpi_ead, "name": "EAD"}],
                }],
            }

    #Speedmeter for lgd
    lgd_options = {
            "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
            "series": [{
                "name": "LGD",
                "type": "gauge",
                "axisLine": {"lineStyle": {"width": 10,},},
                "progress": {"show": "true", "width": 10,},
                "detail": {"valueAnimation": "true", "formatter": "{value}",
                           "textStyle": {"color": "white"},},
                "data": [{"value": kpi_lgd, "name": "LGD"}],
                }],
            }

    dark_theme = {"backgroundColor": "#0E1117", "textStyle": {"color": "white"}}
    # Position speedmeters
    col1, col2, col3 = st.columns(3)
    with col1:
        st_echarts(options=pd_options, width='110%', key=0, theme=dark_theme)
    with col2:
        st_echarts(options=ead_options, width='110%', key=1, theme=dark_theme)
    with col3:
        st_echarts(options=lgd_options, width='110%', key=2, theme=dark_theme)

    #Prescription
    col1, col2 = st.columns(2)
    with col1:
        st.write('Expected Loss(dollars):')
        st.metric(label='Expected Loss', value=kpi_el)
    with col2:
        st.write('An extra type is recommended(dollars):')
        #Determined by the bank
        st.metric(label='Commision to apply', value=kpi_el * 3)
else:
    st.write('Define loan parameters and click on calculate risk')
