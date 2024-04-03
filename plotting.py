import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from addEdge import addEdge, get_color

nodeColor = '#FFCCFF'
nodeSize = 20
lineWidth = 2
lineColor = '#660066'

def make_graph_fig(conn_matrix, feature_names):
    assert conn_matrix.shape[0] == conn_matrix.shape[1]
    N = conn_matrix.shape[0]

    R = 1
    X = R * np.cos(2 * np.pi * np.arange(N) / N)
    Y = R * np.sin(2 * np.pi * np.arange(N) / N)

    edge_x = []
    edge_y = []
    weight_x = []
    weight_y  = []
    weight_values = []
    weight_text = []
    dash_styles = []
    seen_pairs = []

    for edge in np.argwhere(conn_matrix != 0):
        start_node = X[edge[0]], Y[edge[0]]
        end_node = X[edge[1]], Y[edge[1]]
        if start_node != end_node:
            sorted_pair = tuple(np.sort([edge[0], edge[1]])) 
            if sorted_pair in seen_pairs:
                dash_styles.append('dash')
            else:
                dash_styles.append('solid')

            edge_x, edge_y = addEdge(
                start_node, end_node, 
                edge_x, edge_y, 
                lengthFrac=1, arrowPos=0.8, 
                arrowLength=0.04, arrowAngle=30, dotSize=nodeSize,
            )

            weight_x.append(0.8*start_node[0] + 0.2*end_node[0])
            weight_y.append(0.8*start_node[1] + 0.2*end_node[1])
            weight_values.append(conn_matrix[edge[0], edge[1]])
            weight_text.append(
                f'{edge[0]+1} → {edge[1] + 1}, вага: {weight_values[-1]}'
            )
            seen_pairs.append(sorted_pair)
            # print(seen_pairs)

    edge_trace = [] 

    for i in range(len(weight_values)):
        edge_trace.append(
            go.Scatter(
                x=edge_x[9*i:9*(i+1)], y=edge_y[9*i:9*(i+1)],
                line=dict(
                    width=lineWidth, 
                    color=get_color('PuRd', 0.5*weight_values[i]+0.5),
                    dash=dash_styles[i]
                ),
                hoverinfo='none', mode='lines'
            )
        )

        node_trace = go.Scatter(
            x=X, y=Y,
            mode='markers+text',
            text=list(range(1, N+1)),
            textposition='middle center',
            hoverinfo='text', 
            hovertext=feature_names,
            marker=dict(
                showscale=False, 
                color=nodeColor, 
                size=nodeSize,
                line_width=2
            )
        )

        weight_trace = go.Scatter(
            x=weight_x, y=weight_y,
            hoverinfo='text',
            hovertext=weight_text,
            mode='markers',
            marker=dict(
                showscale=False, 
                size=0.0001,
                color=[get_color('RdYlGn', 0.5*w+0.5) for w in weight_values]
            )
        )

    fig = go.Figure(
        data=edge_trace + [node_trace, weight_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=550, height=500,
            margin=dict(b=0,l=50,r=0,t=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )

    return fig


def make_eigval_plot(eigvals):
    reals = np.real(eigvals)
    imags = np.imag(eigvals)
    # radius = np.max(np.abs(eigvals))

    phi = np.linspace(0, 2*np.pi, 100)

    unit_circle = go.Scatter(
        x=np.cos(phi), y=np.sin(phi),
        line=dict(
            width=1, 
            color='black',
            dash='dash'
        ),
        hoverinfo='none', mode='lines'
    )

    large_mask = np.abs(eigvals) > 1
    eigs_small = go.Scatter(
        x=reals[~large_mask], y=imags[~large_mask],
        marker=dict(color='#636EFA'),
        mode='markers',
        hoverinfo='text',
        hovertext=[str(eigv.round(3)) for eigv in eigvals[~large_mask]]
    )

    eigs_large = go.Scatter(
        x=reals[large_mask], y=imags[large_mask],
        marker=dict(color='#EF553B', symbol='x'),
        mode='markers',
        hoverinfo='text',
        hovertext=[str(eigv.round(3)) for eigv in eigvals[large_mask]]
    )

    fig = go.Figure(
        data=[unit_circle, eigs_small, eigs_large],
        layout=go.Layout(
            xaxis=dict(
                showgrid=True, 
                zeroline=False, 
                gridcolor='#bdbdbd', 
                showline=True, 
                showticklabels=True
            ),
            yaxis=dict(
                showgrid=True, 
                zeroline=False, 
                gridcolor='#bdbdbd', 
                showline=True, 
                showticklabels=True, 
                scaleanchor='x', 
                scaleratio=1
            ),
            showlegend=False,
            hovermode='closest',
            width=350, height=300,
            margin=dict(b=0,l=50,r=0,t=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
    )

    return fig


def make_impulse_fig(values, names):
    res_df = pd.DataFrame(values, columns=names)
    impulse_plot_fig = px.line(
        res_df, markers=True, color_discrete_sequence=[
    'rgba(255, 105, 180, 0.7)',    # pink
    'rgba(255, 20, 147, 0.6)',      # deep pink
    'rgba(255, 99, 71, 0.8)',       # tomato
    'rgba(255, 69, 0, 0.5)',        # orange-red
    'rgba(218, 112, 214, 0.9)',     # orchid
    'rgba(220, 20, 60, 0.7)',       # crimson
    'rgba(255, 127, 80, 0.8)',      # coral
    'rgba(255, 192, 203, 0.6)',     # pink
    'rgba(255, 69, 0, 0.7)',        # red-orange
    'rgba(255, 140, 0, 0.7)'        # dark orange
]
    )

    impulse_plot_fig.update_layout(
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            gridcolor='#ffffff', 
            showline=True, 
            showticklabels=True
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            gridcolor='#ffffff', 
            showline=True, 
            showticklabels=True
        ),
        xaxis_title='',
        yaxis_title='',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend_title='фактори',
        legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
),
        width=1000, height=600
    )
    
    return impulse_plot_fig
