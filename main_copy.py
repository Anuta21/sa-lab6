from faulthandler import disable
from io import StringIO
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import numpy as np
from plotting import make_graph_fig, make_eigval_plot, make_impulse_fig
from cognitive import CognitiveMap

st.set_page_config(layout='wide')

st.markdown("""
    <style>
    .stProgress .st-ey {
        background-color: #5fe0de;
    }
    </style>
    """, unsafe_allow_html=True)

st.header('Застосування когнітивного моделювання для розв’язання задач передбачення')
st.write('Виконала **бригада 3 з КА-01** у складі Магаріної Анни, Стожок Анастасії, Захарової Єлизавети')

st.markdown("""
   <style>
            .st-emotion-cache-133e3c0  {background-color: transparent; padding: 0}
            .st-emotion-cache-6qob1r {background-color: #E6AFAF; }
            .main  {background-color: #FEECEF; }
            .st-emotion-cache-18ni7ap  {background-color: #FEECEF; }
            .st-emotion-cache-1fttcpj {display: none}
            .st-emotion-cache-1v7f65g .e1b2p2ww14 {margin: 0}
            .st-emotion-cache-3qrmye {background-color: #E81F64;}
            .st-emotion-cache-16txtl3 {padding: 20px 20px 0px 20px}
            .st-emotion-cache-1629p8f .h1 {padding-bottom: 20px}
            .st-emotion-cache-z5fcl4 {padding: 20px 20px }
}
            
    </style>

    """, unsafe_allow_html=True)

# data_cols = st.columns([2,1,1,1])
# data_file = data_cols[0].file_uploader('Файл вхідних даних', type=['csv'], key='input_file')
# col_sep = data_cols[1].selectbox('Розділювач колонок даних', ('кома', 'пробіл', 'символ табуляції'), key='col_sep')
# dec_sep = data_cols[2].selectbox('Розділювач дробової частини', ('крапка', 'кома'), key='dec_sep')
st.sidebar.title("Дані")
data = st.sidebar.file_uploader('Оберіть файл вхідних даних', type=['csv', 'txt'], key='input_file')
st.sidebar.markdown('<div style="border-bottom: 1px solid #AAA1A5; height: 1px;"/>', unsafe_allow_html=True)
M = []
if data:
    input_file_text = data.getvalue().decode()
    data = StringIO(input_file_text)
    print(input_file_text)
    data = pd.read_csv(data, delimiter=',', decimal='.', index_col=0)

    feature_names = data['назва'].to_list()
    data = data.reset_index().drop(columns=['назва'])

    builder = GridOptionsBuilder.from_dataframe(data)
    builder.configure_default_column(
    resizable=False, filterable=False, editable=True,
    sorteable=False, groupable=False
)
    builder.configure_column('index', header_name='фактор', editable=False)
    builder.configure_grid_options(
    autoSizePadding=1
)
    grid_options = builder.build()

    reload_data = False

    grid_return = AgGrid(
    data, grid_options,
        reload_data=reload_data, 
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.VALUE_CHANGED
    )
    reload_data = False

    df = grid_return['data']
    df_to_save = df.copy()
    df_to_save['назва'] = feature_names
    df_to_save.to_csv('data_edit.csv', index=False)

    M = df.values[:, 1:].astype(float)
    fig = make_graph_fig(M, feature_names)
    st.plotly_chart(fig)
    cogn_map = CognitiveMap(M, df.columns.to_list()[1:])

    st.header('Дослідження на стійкість')

    stab_cols = st.columns(3)
    stab_cols[0].write('Власні числа:')
    for i in cogn_map.getEigenvalues():
        stab_cols[0].write(f"{round(i.real,3)}{'+' if i.imag >= 0 else ''}{round(i.imag,3)}i")

    spectral_radius = cogn_map.getSpectralRadius(cogn_map.getAdjMatrx())
    even_cycles = cogn_map.getEvenCycles()
    even_cycles.sort(key=len, reverse=True)
    number_of_even_cycles = len(even_cycles)
    stab_cols[0].write(f'Спектральний радіус: **{spectral_radius:.5f}**.')
    stab_cols[0].write(f'Структурна стійкість: **' + ('так' if number_of_even_cycles <= 0 else 'ні') + '**.')
    stab_cols[0].write(f'Стійкість за значенням: **' + ('так' if spectral_radius < 1 else 'ні') + '**.')
    stab_cols[0].write(f'Стійкість за збуренням: **' + ('так' if spectral_radius <= 1 else 'ні') + '**.')
    stab_cols[0].write(f'Кількість парних циклів: **{number_of_even_cycles}**.')
    stab_cols[0].write('Список парних циклів:')
    stab_cols[0].text_area(
    '',
    value='\n'.join(
        [' -> '.join([f"V{i+1}" for i in cycle] + [f"V{cycle[0]+1}"]) for cycle in even_cycles]
    ), height=500, disabled=True
)

    st.header('Імпульсне моделювання')
    q = []
    for i in range(M.shape[0]):
        q.append(stab_cols[0].number_input(f'q_{i+1}', min_value=-5.0, max_value=5.0, value=0.0, step=1.0, key=f'q_{i}'))

    iter_count = stab_cols[0].number_input('Кількість ітерацій', min_value=1, max_value=20,value=5, step=1, key='iter_count')

    if st.button('Виконати імпульсне моделювання', key='run_impulse'):
        res = cogn_map.impulse_model(init_q=q, steps=iter_count)
        impulse_plot_fig = make_impulse_fig(res, cogn_map.node_names)
        st.plotly_chart(impulse_plot_fig)