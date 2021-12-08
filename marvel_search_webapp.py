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
## Lines
line_info = pd.read_csv('clean_data/mcu_data_lines.csv', index_col=0).reset_index(drop=True)[['character', 'line', 'movie', 'year']]
line_info.columns = [x.title() for x in line_info.columns.values]
line_info['Relevance'] = np.zeros(len(line_info))

## Characters
char_info = pd.read_csv('clean_data/mcu_data_chars.csv', index_col=0).reset_index(drop=True)
char_info.columns = [x.title() for x in char_info.columns.values]
char_info = char_info.merge(line_info.groupby(['Character'])['Movie'].apply(lambda x: str(list(np.unique(x)))).reset_index(drop=False), on='Character').rename(columns={'Movie':'Movie Apperances'})
char_info['Relevance'] = np.zeros(len(char_info))

## Movies
movie_info = pd.read_csv('clean_data/mcu_data_movies.csv', index_col=0).reset_index(drop=True)
movie_info.columns = ['Movie', 'Script']
movie_info = movie_info.merge(line_info.groupby(['Movie', 'Year']).head(1)[['Movie', 'Year']].reset_index(drop=True), on='Movie', how='left').rename(columns={'Year':'Release Year'})
movie_info['Relevance'] = np.zeros(len(movie_info))
movie_info.loc[movie_info['Movie']=='The Incredible Hulk', 'Release Year'] = 2008

### Model load through URL path:
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

### Create function for using model training
def embed(input):
    return model(input)

### Load Trained Models
## Relevant Lines
trained_model_relevant_lines = tf.saved_model.load('trained_models/relevant_lines')
trained_model_relevant_lines = trained_model_relevant_lines.v.numpy()

## Relevant Chars
trained_model_relevant_chars = tf.saved_model.load('trained_models/relevant_chars')
trained_model_relevant_chars = trained_model_relevant_chars.v.numpy()

## Relevant Movies
trained_model_relevant_movies = tf.saved_model.load('trained_models/relevant_movies')
trained_model_relevant_movies = trained_model_relevant_movies.v.numpy()


### Define Query Function
def SearchDocument(query, topn=10):
    
    #### Universal Processing
    ## Create Query
    q =[query]
    # embed the query for calcluating the similarity
    Q_Train = embed(q)
    
    
    ######### Run Similar Lines #####################
    
    ## Calculate the Similarity
    line_info['Relevance'] = linear_kernel(Q_Train, trained_model_relevant_lines).flatten()
    
    ## Get Positive Relevance Score Only
    relevant_lines = line_info.loc[line_info['Relevance']>0].sort_values(['Relevance'], ascending=False)[['Character', 'Line', 'Movie', 'Relevance']].reset_index(drop=True)
    relevant_lines['Relevance'] = [np.round(x, 3) for x in relevant_lines['Relevance'].values]
    relevant_lines = relevant_lines.head(topn)
        
        
        
    ######### Run Similar Chars #####################
    
    ## Calculate the Similarity
    char_info['Relevance'] = linear_kernel(Q_Train, trained_model_relevant_chars).flatten()
    
    ## Get Positive Relevance Score Only
    relevant_chars = char_info.loc[char_info['Relevance']>0].sort_values(['Relevance'], ascending=False)[['Character', 'Movie Apperances']].reset_index(drop=True)
    relevant_chars = relevant_chars.head(6)
    
    
    ######### Run Similar Movies #####################
    
    ## Calculate the Similarity
    movie_info['Relevance'] = linear_kernel(Q_Train, trained_model_relevant_movies).flatten()
    
    ## Get Positive Relevance Score Only
    relevant_movies = movie_info.loc[movie_info['Relevance']>0].sort_values(['Relevance'], ascending=False)[['Movie', 'Release Year']].reset_index(drop=True)
    relevant_movies = relevant_movies.head(6)
    
    #### Return Results
    
    return relevant_lines, relevant_movies, relevant_chars



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
                    id='movie_used', data=movie_info.sort_values(['Release Year']).reset_index()[['Movie', 'Release Year']].to_dict('records'),
                    columns=[{"name": i, "id": i} for i in movie_info[['Movie', 'Release Year']].columns],
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
                                id='relevant_character_table', data=char_info[['Character', 'Movie Apperances']].to_dict('records'),
                                columns=[{"name": i, "id": i} for i in char_info[['Character', 'Movie Apperances']].columns],
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
                                id='relevant_movies_table', data=movie_info[['Movie', 'Release Year']].to_dict('records'),
                                columns=[{"name": i, "id": i} for i in movie_info[['Movie', 'Release Year']].columns],
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
                            id='relevant_lines_table', data=line_info.to_dict('records'),
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

        relevant_lines_data = line_info.head(10).to_dict('records')
        relevant_movies_data = movie_info.head(5).to_dict('records')
        relevant_char_data = char_info.head(5).to_dict('records')
        
        return default_movie_style, search_results_style, relevant_lines_data, relevant_movies_data, relevant_char_data, 0
    
    

if __name__ == '__main__': 
    app.run_server(debug=True)