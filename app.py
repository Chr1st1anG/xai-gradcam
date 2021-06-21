

import dash_bootstrap_components as dbc
from gradcam import gradcam, extract_predictions
import plotly.graph_objects as go
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from utils import base64_to_img, make_img_graph, byte_png_to_img, resize_img

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# TODO: Show prediction table before images!

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
                multiple=False
            ),
            # html.Button("Predict", id="button-predict",
            #            className="button-main")
        ], className="flexbox"),
        html.Div(
            dash_table.DataTable(
                id='class_table',
                columns=[
                    {'name': 'Class', 'id': 'class'},
                    {'name': 'Confidence', 'id': 'confidence'}
                ],
                row_selectable='single',
                style_cell={'textAlign': 'left', "padding": "0 16px",
                            'font-family': '"Open Sans", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif'},
            ), className="space-top"
        ),
    ], className="container-shadow"),
    html.H4("2. Select convolution block to compute Grad-CAM",
            className="space-top"),
    html.Div([
        html.Div([
            html.H6("EfficientNet B0"),
            html.Img(src=app.get_asset_url(
                'icon_info.svg'), id="tooltip-model"),
            dbc.Tooltip(
                "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua.",
                target="tooltip-model",
                className="tooltip"
            ),
        ], className="header-info"),
        html.Div([
            html.Div([
                html.Div(html.Div("Conv3x3", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv1, 3x3", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 3x3", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 3x3", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 5x5", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 5x5", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 3x3", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 3x3", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 3x3", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 5x5", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 5x5", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 5x5", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 5x5", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 5x5", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 5x5", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 5x5", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("MBConv6, 3x3", className="conv-block-text"),
                         className="conv-block flexbox"),
                html.Div(className="arrow"),
                html.Div(html.Div("Conv1x1", className="conv-block-text"),
                         className="conv-block flexbox"),
            ], className="model-architecture flexbox-row"),
            html.Div(
                dcc.Slider(
                    id="slider_blocks",
                    min=0,
                    max=17,
                    step=None,
                    marks={
                        0: 'stem',
                        1: '1a',
                        2: '2a',
                        3: '2b',
                        4: '3a',
                        5: '3b',
                        6: '4a',
                        7: '4b',
                        8: '4c',
                        9: '5a',
                        10: '5b',
                        11: '5c',
                        12: '6a',
                        13: '6b',
                        14: '6c',
                        15: '6d',
                        16: '7a',
                        17: 'top'
                    },
                    value=17
                ), className="slider",
            ),
            # html.Button("Compute", id="button-gradcam",
            #           className="button-main"),
        ], className="flexbox"
        ),

        html.Div([
            html.Div([
                html.H6("Image"),
                html.Div(dcc.Graph(
                    id="input_graph",
                    figure={},
                    style={'display': 'none'},
                ),
                    id="input-div"),
            ], className="container-img"),
            html.Div([
                html.Div([
                    html.H6("Heatmap"),
                    html.Img(src=app.get_asset_url(
                        'icon_info.svg'), id="tooltip-heatmap"),
                    dbc.Tooltip(
                        "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua.",
                        target="tooltip-heatmap",
                        className="tooltip"
                    ),
                ], className="header-info"),
                html.Div(dcc.Graph(
                    id="heatmap_graph",
                    figure={},
                    style={'display': 'none'},
                ), id="gradcam-div"),
            ], className="container-img"
            ),
        ], className="flexbox-row space-top"),
    ], className="container-shadow"),
], className="container-main flexbox"
)


@app.callback(Output('class_table', 'data'),
              Input('input_graph', 'relayoutData'),
              State('input_graph', 'figure'))
def create_table(relayoutData, figure_dict):
    if figure_dict:
        figure = go.Figure(figure_dict)
        img = figure.to_image(format="png")
        img = byte_png_to_img(img)
        df = extract_predictions(img)
        return df.to_dict('records')


@app.callback(Output('input-div', 'children'),
              Input('upload-image', 'contents'))
def set_input_img(image_str):
    if image_str is not None:
        img = base64_to_img(image_str)
        img = resize_img(img, 600)
        graph = make_img_graph(img, "input_graph", True)
        return graph


@app.callback(Output('gradcam-div', 'children'),
              #Input("button-gradcam", "n_clicks"),
              Input('input_graph', 'figure'),
              Input('class_table', 'selected_rows'),
              Input('slider_blocks', 'value'),
              Input('input_graph', 'relayoutData'))
def update_output(figure_dict, selected_class, slider_value, relayoutData):
    if figure_dict:  # and n_clicks:
        figure = go.Figure(figure_dict)
        img = figure.to_image(format="png")
        img = byte_png_to_img(img)
        img = gradcam(img, selected_class, slider_value)
        graph = make_img_graph(img, "gradcam")
        return graph


if __name__ == '__main__':
    app.run_server(debug=True)
