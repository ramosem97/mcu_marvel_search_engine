### Imports
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.metrics.pairwise import linear_kernel

import dash
from dash import dash_table as dt
from dash import html as html
import plotly.graph_objects as go
from dash import dcc as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output, State

### Read in Data
df = pd.read_csv('clean_data/mcu_data_important_chars.csv', index_col=0).reset_index(drop=True)[['character', 'line', 'movie', 'year']]
df.columns = [x.title() for x in df.columns.values]
df['Relevance'] = np.zeros(len(df))

movie_info = df.groupby(['Movie', 'Year']).head(1)[['Movie', 'Year']].reset_index(drop=True)
movie_info.columns = ['Movie', 'Release Year']

character_info = df.groupby(['Character'])[['Movie']].apply(lambda x: str(list(np.unique(x)))).reset_index(drop=False)
character_info.columns = ['Character', 'Movie Apperances']

### Model load through URL path:
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

### Create function for using model training
def embed(input):
    return model(input)

### Load Trained Model
trained_model = tf.saved_model.load('pretrained_model')
trained_model = trained_model.v.numpy()


### Define Query Function
def SearchDocument(query, topn=10):
    
    ## Create Query
    q =[query]
    
    # embed the query for calcluating the similarity
    Q_Train = embed(q)
    
    # Calculate the Similarity
    df['Relevance'] = linear_kernel(Q_Train, trained_model).flatten()
    
    # Get Positive Relevance Score Only
    similar_lines = df.loc[df['Relevance']>0].sort_values(['Relevance'], ascending=False)[['Character', 'Line', 'Movie', 'Relevance']].reset_index(drop=True)
        
    ## Get Relevant Movies
    relevant_movies = movie_info.set_index(['Movie']).loc[similar_lines.groupby(['Movie'])['Relevance'].max().sort_values(ascending=False).head(5).index.values].reset_index(drop=False)
    
    ## Get Relevant Characters
    relevant_chars = character_info.set_index(['Character']).loc[similar_lines.groupby(['Character'])['Relevance'].max().sort_values(ascending=False).head(5).index.values].reset_index(drop=False)
    
    similar_lines['Relevance'] = [np.round(x, 3) for x in similar_lines['Relevance'].values]
    
    return similar_lines.head(topn), relevant_movies, relevant_chars



app = dash.Dash()

app.layout = html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'Search Engine for Marvel Cinematic Universe - Phase 1', style = {'textAlign':'center',\
                                            'marginTop':40,'marginBottom':40}),
    html.Div([
        
        html.Div([
                
                html.H3( id='QueryLabel', 
                        children = 'Search Here:'),
                
            ], style={'padding-left':'10%',}),
        
        html.Div([ ## Query Text Box and Search Button 
            
            html.Div([dcc.Input( ## Query Text Box
                id='query_content',
                value='',
                placeholder="Enter input",
                type='text',
            )],style={'display': 'inline-block'}),
            
            html.Div([ ## Search Button
                html.Button(
                    'Search',
                    id='search_button',
                    n_clicks=0,
                )], style={'display': 'inline-block'})
            
        ], style={'padding-left':'10%',}),
        
        html.Div(id='DefaultMovieTable', children=[ ## Default Movie Table
                        
            html.Div([
                
                html.H4( id='movie_table_name', 
                        children = 'Phase 1 Movie List'),
                
            ], style={}),
            
            html.Div([ ## Movie List Table
                dt.DataTable(
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    id='movie_used', data=movie_info.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in movie_info.columns],
                    style_cell={'textAlign': 'left'},
                ),
                
            ], style={'width':'50%'}),
            
        ], style={'width':'35%', 'padding-left':'10%'}),
        
        ##### Search Results
        html.Div(id='SearchResults', children=[ ## Search Result Tables
            
            dbc.Row([
                
                dbc.Col([ ##### Relevant Characters
                    html.Div(id='relevant_chars', children=[ ## Relevant Characters table
                        
                        html.Div([ ## Relevant Characters Label
                        
                            html.H4( id='relevant_chars_name', 
                                    children = 'Relevant Characters'),

                        ], style={}),

                        html.Div([ ## Relevant Characters Table
                            dt.DataTable(
                                style_data={
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                },
                                id='relevant_character_table', data=character_info.to_dict('records'),
                                columns=[{"name": i, "id": i} for i in character_info.columns],
                            ),
                        ], style={}),
                        
                    ], style={'display':'inline-block'})
                ]),
            
                dbc.Col([ ##### Relevant Movies
                    html.Div(id='relevant_movies', children=[ ## Relevant Movies table

                        html.Div([ ## Relevant Movies Label

                            html.H4( id='relevant_movies_name', 
                                    children = 'Relevant Movies'),

                        ], style={}),

                        html.Div([ ## Relevant Movies Table
                            dt.DataTable(
                                style_data={
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                },
                                id='relevant_movies_table', data=movie_info.to_dict('records'),
                                columns=[{"name": i, "id": i} for i in movie_info.columns],
                            ),
                        ], style={}),

                    ], style={'display':'inline-block'})
                ]),
            ]),
            
            dbc.Row([ ##### Relevant Lines
                html.Div(id='relevant_lines', children=[ ## Relevant line table

                    html.Div([ ## 

                        html.H4( id='relevant_line_name', 
                                children = 'Relevant Movie Lines'),

                    ], style={}),

                    html.Div([ ## Relevant line Table
                        dt.DataTable(
                            style_data={
                                'whiteSpace': 'normal',
                                'height': 'auto',
                            },
                            id='relevant_lines_table', data=df.to_dict('records'),
                            columns=[{"name": i, "id": i} for i in ['Character', 'Line', 'Movie', 'Relevance']],
                        ),
                    ], style={}),

                ], style={'display':'inline-block'}),
            ]),
            
        ], style={'width':'80%','padding-left':'10%', 'padding-right':'10%', 'display':'block'}),
        
    ])

])
    
    
@app.callback([Output(component_id='DefaultMovieTable', component_property= 'style'),
              Output(component_id='SearchResults', component_property= 'style'),
              Output(component_id='relevant_lines_table', component_property= 'data'),
              Output(component_id='relevant_movies_table', component_property= 'data'),
              Output(component_id='relevant_character_table', component_property= 'data'),
              Output(component_id='search_button', component_property= 'n_clicks')],
              Input(component_id='search_button', component_property= 'n_clicks'),
              State('query_content', 'value'))

def search_results(submitted, query):
    
    if (submitted & (query!= '')):
        
        query_res = SearchDocument(query, topn=10)
        
        default_movie_style = {'width':'80%','padding-left':'10%', 'padding-right':'10%', 'display':'none'}
        search_results_style = {'width':'80%','padding-left':'10%', 'padding-right':'10%', 'display':'block'}
        
        query_res[0]['Relevance'] = [np.round(x, 3) for x in query_res[0]['Relevance'].values]
        
        relevant_lines_data = query_res[0].to_dict('records')
        relevant_movies_data = query_res[1].head(5).to_dict('records')
        relevant_char_data = query_res[2].head(5).to_dict('records')

        return default_movie_style, search_results_style, relevant_lines_data, relevant_movies_data, relevant_char_data, 0
    
    else:
        
        default_movie_style = {'width':'80%','padding-left':'10%', 'padding-right':'10%', 'display':'block'}
        search_results_style = {'width':'80%','padding-left':'10%', 'padding-right':'10%', 'display':'none'}

        relevant_lines_data = df.head(10).to_dict('records')
        relevant_movies_data = movie_info.head(5).to_dict('records')
        relevant_char_data = character_info.head(5).to_dict('records')
        
        return default_movie_style, search_results_style, relevant_lines_data, relevant_movies_data, relevant_char_data, 0
    
    

if __name__ == '__main__': 
    app.run_server(debug=True)