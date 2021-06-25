import dash_bootstrap_components as dbc
from dash_html_components.P import P
from gradcam import gradcam, extract_predictions
import plotly.graph_objects as go
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from utils import base64_to_img, make_img_graph, byte_png_to_img, resize_img

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Visual Analytics"

server = app.server

app.layout = html.Div([
    html.H4("1. Upload an image to get started with Grad-CAM"),
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
        ], className="flexbox"),
    ], className="container-shadow"),
    html.H4("2. Perturbate your image to influence the model's prediction",
            className="space-top"),
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H6("Image"),
                    html.Img(src=app.get_asset_url(
                        'icon_info.svg'), id="tooltip-image"),
                    dbc.Tooltip(
                        "Draw rectangles or other shapes into the image to hide it from the network. Learn more about perturbation below.",
                        target="tooltip-image",

                    ),
                ], className="header-info"),
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
                        "Here the area of the image that caused the network´s decision is highlighted. Learn more about the Grad-CAM algorithm below.",
                        target="tooltip-heatmap",

                    ),
                ], className="header-info"),
                html.Div(dcc.Graph(
                    id="heatmap_graph",
                    figure={},
                    style={'display': 'none'},
                ), id="gradcam-div"),
            ], className="container-img"
            ),
        ], className="flexbox-row"),
        html.Div([
            html.Div([
                html.H6("Predictions"),
                html.Img(src=app.get_asset_url(
                    'icon_info.svg'), id="tooltip-prediction"),
                dbc.Tooltip(
                    "Select a particular class to calculate Grad-CAM with, otherwise the class with the highest confidence is chosen.",
                    target="tooltip-prediction",

                ),
            ], className="header-info"),
            dash_table.DataTable(
                id='class_table',
                columns=[
                    {'name': 'Class', 'id': 'class'},
                    {'name': 'Confidence', 'id': 'confidence'}
                ],
                row_selectable='single',
                style_cell={'textAlign': 'left', "padding": "0 16px",
                            'font-family': '"Open Sans", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif'},
            )], className="space-top"
        ),
        html.Details([
            html.Summary(
                html.Div([
                    html.H6("EfficientNet B0"),
                    html.Img(src=app.get_asset_url(
                        'icon_info.svg'), id="tooltip-model"),
                    dbc.Tooltip(
                        "Select the feature map of the network that will be used for the Grad-CAM calculation.",
                        target="tooltip-model",
                    ),
                ], className="header-info div-summary"),
            ),
            html.Div([
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
            ], className="flexbox"
            ),
        ], className="space-top"),
    ], className="container-shadow"),
    html.Details([
        html.Summary(html.H4("How does it work?"),
                     style={"margin-bottom": "12px"}),
        html.Div([
            html.H5("Grad-CAM"),
            html.P("Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet."),
            html.H5("Perturbation"),
            html.P("Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.")
        ], className="container-shadow"),
    ], className="space-top"),
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
