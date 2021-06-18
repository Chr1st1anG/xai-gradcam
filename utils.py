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



def byte_png_to_img(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    return img



def make_img_graph(img, id):
    fig = go.Figure()

    # Constants
    img_width = 224
    img_height = 224
    scale_factor = 2


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
        'modeBarButtonsToRemove': ['autoScale2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'zoom2d', 'lasso2d'],
        'modeBarButtonsToAdd': ['drawrect', 'drawclosedpath','eraseshape']}

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    fig.update_layout(dragmode='drawrect',
                      # style of new shapes
                      newshape=dict(line_color='black',
                                    fillcolor='black',
                                    opacity=1))

    fig.update_layout(dragmode='drawclosedpath',
                      # style of new shapes
                      newshape=dict(line_color='black',
                                    fillcolor='black',
                                    opacity=1))

    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    graph = dcc.Graph(
        id=id,
        figure=fig,
        config=config,
        style={
            "width": "100%",
            "height": "auto",
            "border-radius": "5px"
        })

    return graph
