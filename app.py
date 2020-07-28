import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from model_utils import Embedding
from utils import read_files, similarity_matrix, sort_matrix
import time

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.title = "DIF"

app.layout = html.Div(children=[
    html.Div(children="Duplicate Image Finder ",
             id="headline"),
    html.Br(),
    html.Div([dcc.Input(
        id="input_path",
        type="text",
        placeholder="Paste dataset folder path",
        debounce=True,)
    ], id='input_path_div'),
    html.Br(),
    dcc.Loading(
        id="loading-index",
        children=[html.Div(id="loading-index-2")],
        type="circle",
    ),
    # html.Div(id='file_len_output'),


])


@app.callback(Output('loading-index-2', 'children'),
              [Input('input_path', 'value')],
              [State('loading-index-2', 'children')])
def get_path(path, children):
    files = read_files(path)
    if path:
        children.append(html.P(f"files found : {len(files)}"))
        embedding = Embedding(files)
        embs = np.array(embedding.embeddings())
        matrix = similarity_matrix(embs, embs)
        index_pair = sort_matrix(matrix)
        np.save('index_pair.npy', index_pair)
        return children
    return []

# @app.callback(Output('display_images', 'children'),
#               [Input('input_path', 'value')])
# def get_path(path):
#     files = read_files(path)
#     return html.P(f"files found : <strong>{len(files)}</strong>")


if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")
