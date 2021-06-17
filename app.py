from dash_html_components.A import A
from dash_html_components.Div import Div
from dash_html_components.H3 import H3
from dash_html_components.P import P
from pandas.io.formats import style
from gradcam import gradcam, extract_predictions

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from utils import base64_to_img, make_img_graph

import pandas as pd
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Example: Get data of selected row
# https://community.plotly.com/t/how-to-get-data-of-selected-rows-from-dash-datatable/32383

server = app.server

app.layout = html.Div([
    html.H4("1. Upload and predict image"),
    html.Div([
        html.Div([
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                className="upload-form",
                # Allow multiple files to be uploaded
                multiple=False
            ),
            html.Button("Predict", id="button-predict",
                        className="button-main")
        ], className="flexbox-row"),
        html.Div(
            dash_table.DataTable(
                id='class_table',
                columns=[
                    {'name': 'Class', 'id': 'class'},
                    {'name': 'Confidence', 'id': 'confidence'}
                ],
                row_selectable='single',
                style_cell={'textAlign': 'left', "padding": "0 16px"},
            ), className="space-top"
        ),
    ], className="container-shadow"),
    html.H4("2. Select convolution block to compute Grad-CAM",
            className="space-top"),
    html.Div([
        html.Div([
            html.Div(
                dcc.Slider(
                    id="slider_blocks",
                    min=0,
                    max=4,
                    step=None,
                    marks={
                        0: 'Conv 1',
                        1: 'Conv 2',
                        2: 'Conv 3',
                        3: 'Conv 4',
                        4: 'Conv 5'
                    },
                    value=4
                ), className="slider",
            ),
            html.Button("Compute", id="button-gradcam",
                        className="button-main"),
        ], className="flexbox-row"
        ),

        html.Div([
            html.Div([
                html.H6("Image"),
                html.Div(id="input-div"),
            ], className="container-img"),
            html.Div([
                html.H6("Heatmap"),
                html.Div(id="gradcam-div"),
            ], className="container-img"
            ),
        ], className="flexbox-row space-top"),
    ], className="container-shadow"),
    dcc.Store(id='img')
], className="container-main flexbox"
)


@app.callback(Output('class_table', 'data'), Input('img', 'data'))
def create_table(image_str):
    if image_str is not None:
        img = base64_to_img(image_str)
        df = extract_predictions(img)
        return df.to_dict('records')


@ app.callback(Output('img', 'data'), Input('upload-image', 'contents'))
def img_to_dcc(image_str):
    return image_str


@ app.callback(Output('input-div', 'children'),
               Input('img', 'data'))
def set_input_img(image_str):
    if image_str is not None:
        img = base64_to_img(image_str)
        graph = make_img_graph(img, "input")
        return graph


@ app.callback(Output('gradcam-div', 'children'),
               Input('img', 'data'))
def update_output(image_str):
    if image_str is not None:
        img = base64_to_img(image_str)
        img = gradcam(img)
        graph = make_img_graph(img, "gradcam")
        return graph


if __name__ == '__main__':
    app.run_server(debug=True)
