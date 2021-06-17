from gradcam import gradcam, extract_predictions

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

from utils import base64_to_img, make_img_graph

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div([
        html.Div(id="input-div", className="six columns"),
        html.Div(id="gradcam-div", className="six columns")],
        className="row"),

    dash_table.DataTable(
        id='class_table',
        columns=[
            {'name': 'class', 'id': 'class'},
            {'name': 'confidence', 'id': 'confidence'}
        ],
        row_selectable='single',  # get selected with State('class_table', 'selected_rows')
        style_cell=dict(textAlign='left'),
        style_header=dict(backgroundColor="paleturquoise"),
        style_data=dict(backgroundColor="lavender")
    ),

    dcc.Store(id='img')
])


@app.callback(Output('class_table', 'data'), Input('img', 'data'))
def create_table(image_str):
    if image_str is not None:
        img = base64_to_img(image_str)
        df = extract_predictions(img)
        return df.to_dict('records')


@app.callback(Output('img', 'data'), Input('upload-image', 'contents'))
def img_to_dcc(image_str):
    return image_str


@app.callback(Output('input-div', 'children'),
              Input('img', 'data'))
def set_input_img(image_str):
    if image_str is not None:
        img = base64_to_img(image_str)
        graph = make_img_graph(img, "input")
        return graph


@app.callback(Output('gradcam-div', 'children'),
              Input('img', 'data'))
def update_output(image_str):
    if image_str is not None:
        img = base64_to_img(image_str)
        img = gradcam(img)
        graph = make_img_graph(img, "gradcam")
        return graph


if __name__ == '__main__':
    app.run_server(debug=True)
