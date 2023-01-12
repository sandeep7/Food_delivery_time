from os import name
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly as py
import plotly.offline as pyo
from plotly.subplots import make_subplots
from collections import Counter
from math import ceil


class Plot:
    '''
    Class to create histogram plots for better visualisation
    '''

    @staticmethod
    def PlotHistogram(data,feature_list, cap_bins = False):
        '''
        Plot the histograms for the above variables
        '''
        rows = ceil(len(feature_list)/2)
        cols =  2 if len(feature_list) > 1 else 1

        subplot_titles = ["Histogram of " + feature.replace("_"," ") for feature in feature_list]

        fig = make_subplots(rows=rows,
                            cols=cols,
                            subplot_titles=subplot_titles)
        
        n_r = 1
        n_c = 1
        for feature in feature_list:
            fig.add_trace(
                go.Histogram(x=data[feature],
                 histnorm='percent',
                 name = feature
                ), row=n_r, col=n_c)

            # Update xaxis properties
            fig.update_xaxes(title_text=feature, row=n_r, col=n_c)
            # Update yaxis properties
            fig.update_yaxes(title_text="%", row=n_r, col=n_c)

            if n_c == 2:
                n_r += 1
                n_c = 1
            else:
                n_c += 1
        
        fig.update_layout(height=500*rows,bargap=0.2)

        fig.show()


    @staticmethod
    def PlotBar(data,feature_list):
        '''
        Plot the histograms for the above variables
        '''
        rows = ceil(len(feature_list)/2)
        cols =  2 if len(feature_list) > 1 else 1

        subplot_titles = ["BarPlot of " + feature.replace("_"," ") for feature in feature_list]

        fig = make_subplots(rows=rows,
                            cols=cols,
                            subplot_titles=subplot_titles)
        
        n_r = 1
        n_c = 1
        for feature in feature_list:
            d = data[feature].value_counts(normalize = True).reset_index()
            d['count'] = data[feature].value_counts().reset_index()[feature]
            fig.add_trace(
                go.Bar(x=d['index'],
                       y=d[feature],
                       text = d['count'],
                       name = feature), row=n_r, col=n_c)

            # Update xaxis properties
            fig.update_xaxes(title_text=feature, row=n_r, col=n_c)
            # Update yaxis properties
            fig.update_yaxes(title_text="%", row=n_r, col=n_c)

            if n_c == 2:
                n_r += 1
                n_c = 1
            else:
                n_c += 1
        
        fig.update_layout(height=500*rows,bargap=0.2)

        fig.show()

    @staticmethod
    def PlotBarWithPrepTime(data,feature_list,sort_values = False):
        '''
        Plot the histograms for the above variables with Median Prep Time to understand importance 
        '''
        rows = ceil(len(feature_list)/2)
        cols =  2 if len(feature_list) > 1 else 1

        subplot_titles = ["Bar Plot of " + feature.replace("_"," ") +" With Median Prep Time" for feature in feature_list]

        fig = make_subplots(rows=rows,
                            cols=cols,
                            subplot_titles=subplot_titles,
                            specs=[[{"secondary_y": True}]*cols]*rows)
        
        n_r = 1
        n_c = 1
        for feature in feature_list:
            d = data[feature].value_counts(normalize = True).reset_index()
            d['count'] = data[feature].value_counts().reset_index()[feature]
            d['prep_time_seconds'] = d.merge(data.groupby(feature)['prep_time_seconds'].median().reset_index(),left_on=['index'],right_on=[feature],how = 'inner')['prep_time_seconds']
            if sort_values:
                d = d.sort_values('index')
            fig.add_trace(
                go.Bar(x=d['index'],
                       y=d[feature],
                       text = d['count'],
                       name = feature), row=n_r, col=n_c)
            # Update xaxis properties
            fig.update_xaxes(title_text=feature, row=n_r, col=n_c)
            # Update yaxis properties
            fig.update_yaxes(title_text="%", secondary_y=False,row=n_r, col=n_c)
            # Update yaxis properties
            fig.update_yaxes(title_text="Median Prep Time Sec",secondary_y=True,row=n_r, col=n_c)

            fig.add_trace(
                go.Scatter(x=d["index"], y=d["prep_time_seconds"], name="Prep time Seconds"),
                secondary_y=True,row=n_r, col=n_c)

            if n_c == 2:
                n_r += 1
                n_c = 1
            else:
                n_c += 1
        
        fig.update_layout(height=500*rows,bargap=0.2)

        fig.show()