import io
import base64

from PIL import Image
import dash_core_components as dcc
import plotly.graph_objects as go

def img_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = "data:image/png;base64," + img_str.decode()
    return img_str


def base64_to_img(img_string):
    img = Image.open(io.BytesIO(base64.b64decode(img_string.split(',')[1])))
    return img


def make_gradcam_graph(img):
    fig = go.Figure()

    # Constants
    img_width = 224
    img_height = 224
    scale_factor = 3

    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img)
    )

    config = {
        'modeBarButtonsToRemove': ['autoScale2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'zoom2d', 'resetScale2d', 'lasso2d',
                                   'toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian']}

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    graph = dcc.Graph(
        id='gradcam',
        figure=fig,
        config=config)

    return graph