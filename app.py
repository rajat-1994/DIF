import dash
import base64
import numpy as np
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from model_utils import Embedding
from utils import similarity_matrix, sort_matrix
from utils import read_files, new_df, save_df

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.DARKLY]

SESSION_ID = 0

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.title = "DIF"


app.layout = html.Div(children=[
    html.Div(children="Duplicate Image Finder ",
             id="headline"),
    html.Br(),
    dbc.Row([dbc.Input(
        id="input_path",
        type="text",
        bs_size="lg",
        placeholder="Paste dataset folder path ",
        debounce=True,)
    ],
        justify="center",
        id='input_path_div'),

    html.Br(),
    html.Br(),

    dcc.Loading(
        id="loader",
        children=[html.Div(id="loading-div")],
        type="dot",
    ),

    html.Br(),
    # html.Div([dbc.Button("Start", color="success", n_clicks=0,
    #                      size='lg', id='start-button'),
    #           ],
    #          id='start-row'),
    dbc.Row(
        [dbc.Col(id='col-card-1', width=4),
         dbc.Col(id='col-card-2', width=4)
         ],
        justify="center",
        id='display_layout'
    )
])


# @ app.callback([Output('start-row', 'children'), ],
#                [Input('loading-div', 'children')],)
# def remove_start_buttom(children):
#     print("remove clicks", children)
#     return dbc.Button("Start", color="success", n_clicks=0,
#                       size='lg', id='start-button')


@ app.callback([Output('col-card-1', 'children'),
                Output('col-card-2', 'children'), ],
               [Input('loading-div', 'children'),
                Input('display_layout', 'children')])
def display_image(children, dl_children):
    global SESSION_ID
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print("changed_id", changed_id)
    print("clicks", children)
    print("dl child", dl_children)
    if children != []:
        # Reading files
        index_pair = np.load('index_pair.npy')
        files_path = pd.read_csv('./files.csv')
        # Checking which button is clicked and update Session id
        update_session(index_pair, files_path, changed_id)
        # Reading images and filename
        image1, image2, filename1, filename2 = encoded_images(
            index_pair, files_path)
        # Saving updated files datafrmae
        save_df(files_path)

        top_card = dbc.Card(
            [
                dbc.CardImg(src=image1, id='img1', top=True),
                dbc.CardBody([
                    dbc.Row(html.P(filename1, id='filename-1'),
                            justify="center"),
                    dbc.Button('DELETE',
                               color="danger",
                               size='lg',
                               id="button1",
                               n_clicks=0,)
                ]),
            ],
            style={"width": "auto"},
        )

        bottom_card = dbc.Card(
            [
                dbc.CardImg(src=image2, id='img2', top=True),
                dbc.CardBody([
                    dbc.Row(html.P(filename2, id='filename-2'),
                            justify="center"),
                    dbc.Button('DELETE',
                               color="danger",
                               size='lg',
                               id="button2",
                               n_clicks=0,)
                ]),
            ],
            style={"width": "auto"},
        )
        return top_card, bottom_card
    return "", ""


def update_session(index_pair, files_df, changed_id):
    global SESSION_ID
    idx1, idx2 = index_pair[SESSION_ID]
    if 'button1' in changed_id:
        files_df.is_deleted.iloc[idx1] = 1
        SESSION_ID += 1
    elif 'button2' in changed_id:
        files_df.is_deleted.iloc[idx2] = 1
        SESSION_ID += 1


def encoded_images(index_pair, files_df):
    global SESSION_ID
    idx1, idx2 = index_pair[SESSION_ID]
    if files_df.is_deleted.iloc[idx1] == 0 and files_df.is_deleted.iloc[idx2] == 0:
        file1, file2 = files_df.files.iloc[idx1], files_df.files.iloc[idx2]
        # Reading Images and encoding them as base64
        enc_img1 = base64.b64encode(open(file1, 'rb').read())
        enc_img2 = base64.b64encode(open(file2, 'rb').read())
        enc_images = [
            f"data:image/png;base64,{enc_img1.decode()}",
            f"data:image/png;base64,{enc_img2.decode()}"]
        # Sending file names
        filenames = [file1.split('/')[-1], file2.split('/')[-1]]
        return enc_images+filenames
    else:
        SESSION_ID += 1
        return encoded_images(index_pair, files_df)
    return ['', '', '', '']


@ app.callback(Output('loading-div', 'children'),
               [Input('input_path', 'value')],
               [State('loading-div', 'children')])
def get_path(path, children):
    files = read_files(path)
    files = files[:100]
    if path:
        children.append(html.P(f"files found : {len(files)}"))
        embedding = Embedding()
        embs = np.array(embedding.embeddings(files))
        matrix = similarity_matrix(embs, embs)
        index_pair = sort_matrix(matrix)
        np.save('index_pair.npy', index_pair)
        df = new_df(files)
        save_df(df, './files.csv')
        return children
    return []


if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")
