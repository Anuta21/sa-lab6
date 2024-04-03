from faulthandler import disable
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import numpy as np
from plotting import make_graph_fig, make_eigval_plot, make_impulse_fig
from cognitive import CognitiveMap

st.set_page_config(
    page_title='СА ЛР6', 
    page_icon='📈',
    layout='wide',
    menu_items={
        'About': 'Лабораторна робота №6 з системного аналізу. Виконали бригади 1 та 2 з КА-81: Галганов Олексій, Єрко Андрій, Фордуй Нікіта, Билим Кирило, Підвальна Анна, Яковина Андрій.'
    })

st.title('Застосування когнітивного моделювання для розв’язання задач передбачення')

# data_cols = st.columns([2,1,1,1])
# data_file = data_cols[0].file_uploader('Файл вхідних даних', type=['csv'], key='input_file')
# col_sep = data_cols[1].selectbox('Розділювач колонок даних', ('кома', 'пробіл', 'символ табуляції'), key='col_sep')
# dec_sep = data_cols[2].selectbox('Розділювач дробової частини', ('крапка', 'кома'), key='dec_sep')

data = pd.read_csv('data.csv', delimiter=',', decimal='.', index_col=0)

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

col1, col2 = st.columns(spec=[2.5, 2])
with col1:
    reload_data = False
    if col1.button('Скинути зміни в матриці', key='reset_edit'):
        reload_data = True

    grid_return = AgGrid(
        data, grid_options,
        reload_data=reload_data, 
        update_mode=GridUpdateMode.VALUE_CHANGED
    )
    reload_data = False

    df = grid_return['data']
    df_to_save = df.copy()
    df_to_save['назва'] = feature_names
    df_to_save.to_csv('data_edit.csv', index=False)

with col2:
    M = df.values[:, 1:].astype(float)
    fig = make_graph_fig(M, feature_names)
    col2.plotly_chart(fig)
    cogn_map = CognitiveMap(M, df.columns.to_list()[1:])

st.header('Перевірка на стійкість')

stab_cols = st.columns(3)
stab_cols[0].write('Власні числа:')
stab_cols[0].plotly_chart(make_eigval_plot(cogn_map.getEigenvalues()))

spectral_radius = cogn_map.getSpectralRadius(cogn_map.getAdjMatrx())
even_cycles = cogn_map.getEvenCycles()
even_cycles.sort(key=len, reverse=True)
number_of_even_cycles = len(even_cycles)
stab_cols[1].write(f'Спектральний радіус $R$: **{spectral_radius:.5f}**.')
stab_cols[1].write(f'Стійкість за значенням ($R < 1$): **' + ('так' if spectral_radius < 1 else 'ні') + '**.')
stab_cols[1].write(f'Стійкість за збуренням ($R \leq 1$): **' + ('так' if spectral_radius <= 1 else 'ні') + '**.')
stab_cols[1].write(f'Кількість парних циклів: **{number_of_even_cycles}**.')
stab_cols[2].write('Список парних циклів:')
stab_cols[2].text_area(
    '',
    value='\n'.join(
        [' → '.join([str(i+1) for i in cycle] + [str(cycle[0]+1)]) for cycle in even_cycles]
    ), height=300, disabled=True
)

st.header('Імпульсне моделювання')
impulse_cols = st.columns([1, 4])
q = []
for i in range(M.shape[0]):
    q.append(
        impulse_cols[0].number_input(f'q_{i+1}', 
        min_value=-5.0, max_value=5.0, value=0.0,
        step=1.0, key=f'q_{i}')
    )

iter_count = impulse_cols[1].number_input(
    'Кількість ітерацій', min_value=1, max_value=20,
    value=5, step=1, key='iter_count'
)

if impulse_cols[1].button('Запустити моделювання', key='run_impulse'):
    res = cogn_map.impulse_model(init_q=q, steps=iter_count)
    impulse_plot_fig = make_impulse_fig(res, cogn_map.node_names)
    impulse_cols[1].plotly_chart(impulse_plot_fig)