import datetime

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
#import dash_bootstrap_components as dbc
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

from PIL import Image
import plotly.express as px

import cv2 as cv
import io
import base64
import numpy as np
## init model
import torch
import clip
from PIL import Image
from focusing_check import image_feat, px_visualise
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



app = Dash(__name__)#,external_stylesheets=[dbc.themes.BOOTSTRAP]
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
        multiple=True
    ),
    html.Div(id='output-image-upload'),
#     dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
#     dbc.Progress(id="progress"),
    html.Label('Text Input'),
    html.Div(),
    dcc.Input(id='text1',value='', type='text'),
    html.Div(),
    #html.Label('Text Input'),
    dcc.Input(id='text2',value='', type='text'),
    html.Div(),
    dcc.Input(id='text3',value='', type='text'),
    html.Div(),
    dcc.Input(id='text4',value='', type='text'),
    html.Div(),
    dcc.Input(id='text5',value='', type='text'),
    html.Div(),
    dcc.Input(id='text6',value='', type='text'),
    html.Div(),
    dcc.Input(id='text7',value='', type='text'),
    
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    dcc.Graph(id='result'),
    dcc.Graph(id='focus'),
])

# @app.callback(
#     [Output("progress", "value"), Output("progress", "label")],
#     [Input("progress-interval", "n_intervals")],
# )
# def update_progress(n):
#     # check progress of some background process, in this example we'll just
#     # use n_intervals constrained to be in 0-100
#     progress = min(n % 110, 100)
#     # only add text after 5% progress to ensure text isn't squashed too much
#     return progress, f"{progress} %" if progress >= 5 else ""

def parse_contents(contents, filename, date):
    #print(type(contents))
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        # html.Img(src=contents),
        dcc.Graph(id='life-exp-vs-gdp', figure=update_fig(contents)),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])
def update_fig(contents):
    #img = Image.open(filename)

    #base64_decoded = base64.b64decode()

    image = decode_img(contents)
    fig = px.imshow(image)
    return fig

def decode_img(contents):
    img_uri = contents
    encoded_img = img_uri.split(",")[1]
    binary = base64.b64decode(encoded_img)
    with open("test_save.png", 'wb') as f:
        f.write(binary)
    image = Image.open("test_save.png")
    image = np.array(image)

    #image = Image.open(io.BytesIO(base64_decoded))
    #image = cv.imdecode(image, cv.IMREAD_COLOR)

    return image

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]


        return children

@app.callback(
    Output('result', 'figure'),
    Output('focus', 'figure'),
    Input('submit-button-state', 'n_clicks'),
    State('text1', 'value'),
    State('text2', 'value'),
    State('text3', 'value'),
    State('text4', 'value'),
    State('text5', 'value'),
    State('text6', 'value'),
    State('text7', 'value'),
    State('upload-image', 'contents'))
def update_result(n_clicks, text1, text2, text3, text4, text5, text6, text7, contents):
    if n_clicks == 0:
        return
    #print(type(contents[0]))
    #filename = filename[0]
    
    #print(img.shape)
    #image = preprocess(img).unsqueeze(0).to(device)
    #print(image.shape)
    raw_image = Image.open("test_save.png")
    feat ,image = image_feat(raw_image)
    list_text = list(set([text1,text2,text3,text4,text5,text6,text7]))
    fig2,probs = px_visualise(list_text, feat,image)
    
    #image = preprocess(Image.open('test_save.png')).unsqueeze(0).to(device)

    #print(Image.open(filename).shape)
#     text = clip.tokenize(list_text).to(device)


#     with torch.no_grad():
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)
        
#         logits_per_image, logits_per_text = model(image, text)
#         probs = logits_per_image.softmax(dim=-1).cpu().numpy()*100
    fig = px.histogram(x = probs[0],y= list_text)
    print(probs[0])
    fig.update_xaxes()
    #print(probs)
    return [fig, fig2]

if __name__ == '__main__':
    app.run_server(debug=True, host='192.168.0.17')