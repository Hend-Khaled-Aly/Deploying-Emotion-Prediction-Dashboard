import dash
import pandas as pd
import plotly.express as px
import joblib
import base64
import os
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Load data and model
df = pd.read_csv("data/social_media_emotions.csv")
model = joblib.load("rf_pipeline.pkl")

def create_figure_1(df):
    counts = df.groupby(['Platform', 'Dominant_Emotion']).size().reset_index(name='Count')

    # Get top 3 emotions per platform
    top_emotions_per_platform = (
        counts.sort_values(['Platform', 'Count'], ascending=[True, False])
        .groupby('Platform')
        .head(3)
    )

    # Unique platforms
    platforms = top_emotions_per_platform['Platform'].unique()

    # Create subplots
    fig = make_subplots(rows=1, cols=len(platforms), subplot_titles=platforms)

    # Add bars for each platform
    for i, platform in enumerate(platforms):
        platform_data = top_emotions_per_platform[top_emotions_per_platform['Platform'] == platform]
        fig.add_trace(
            go.Bar(
                x=platform_data['Dominant_Emotion'],
                y=platform_data['Count'],
                showlegend=False
            ),
            row=1, col=i+1
        )
        fig.update_xaxes(title_text="Emotion", tickvals=platform_data['Dominant_Emotion'], row=1, col=i+1)
        if i == 0:
            fig.update_yaxes(title_text="Number of Users", row=1, col=1)

    # Set layout
    fig.update_layout(
        title={
            'text': "<b>Top 3 Dominant Emotions by Social Media Platform</b>",
            'font': {'size': 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=400,
        width=1200,
        margin=dict(t=80)
    )

    return fig


def create_figure_2(df):
    # Step 1: Define emojis and colors for emotions
    emotion_info = {
        'Happiness': {'emoji': '😊', 'color': '#FFD700'},
        'Neutral': {'emoji': '😐', 'color': '#A9A9A9'},
        'Anxiety': {'emoji': '😰', 'color': '#FF7F50'},
        'Sadness': {'emoji': '😢', 'color': '#1E90FF'},
        'Boredom': {'emoji': '🥱', 'color': '#C0C0C0'},
        'Anger': {'emoji': '😠', 'color': '#DC143C'}
    }

    # Step 2: Create time bins
    bins = [0, 30, 60, 90, 120, 150, 180, df['Daily_Usage_Time (minutes)'].max()]
    labels = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180+']
    df['Time_Range'] = pd.cut(df['Daily_Usage_Time (minutes)'], bins=bins, labels=labels, right=False)

    # Step 3: Group and calculate percentages
    grouped = df.groupby(['Time_Range', 'Dominant_Emotion']).size().reset_index(name='Count')
    total_per_range = grouped.groupby('Time_Range')['Count'].transform('sum')
    grouped['Percentage'] = (grouped['Count'] / total_per_range) * 100

    # Step 4: Add emojis and colors
    grouped['Labeled_Emotion'] = grouped['Dominant_Emotion'].apply(lambda emo: f"{emotion_info[emo]['emoji']} {emo}")
    grouped['Color'] = grouped['Dominant_Emotion'].apply(lambda emo: emotion_info[emo]['color'])

    # Step 5: Build plot with dropdown
    fig = go.Figure()
    time_ranges = grouped['Time_Range'].unique().tolist()

    for time_range in time_ranges:
        df_range = grouped[grouped['Time_Range'] == time_range]
        fig.add_trace(go.Bar(
            x=df_range['Labeled_Emotion'],
            y=df_range['Percentage'],
            name=str(time_range),
            visible=(time_range == '30-60'),
            text=df_range['Percentage'].round(1).astype(str) + '%',
            textposition='auto',
            marker_color=df_range['Color']
        ))

    # Step 6: Dropdown menu
    dropdown_buttons = [
        dict(label=str(r),
             method='update',
             args=[{'visible': [r == tr for tr in time_ranges]},
                   {'title': {
                       'text': f"<b>Dominant Emotions ({r})</b>",
                       'font': {'size': 24},
                       'x': 0.5,
                       'xanchor': 'center'
                   }}])
        for r in time_ranges
    ]

    fig.update_layout(
        title={
            'text': "<b>Dominant Emotions (30–60 min)</b>",
            'font': {'size': 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Dominant Emotion",
        yaxis_title="Percentage of Users",
        updatemenus=[dict(
            active=time_ranges.index('30-60') if '30-60' in time_ranges else 0,
            buttons=dropdown_buttons,
            x=0.5,
            xanchor="center",
            y=1.15,
            yanchor="top"
        )],
        plot_bgcolor='white',
        height=500,
        width=800
    )

    return fig

def create_figure_3(df):
    # Compute average posts per day by Platform and Gender
    avg_posts = df.groupby(['Platform', 'Gender'])['Posts_Per_Day'].mean().reset_index()

    # Ensure 'Male' is the default visible gender
    genders = ['Male'] + [g for g in avg_posts['Gender'].unique() if g != 'Male']
    platforms = avg_posts['Platform'].unique()

    # Define colors for genders
    gender_colors = {
        'Male': 'steelblue',
        'Female': 'mediumvioletred',
        'Other': 'seagreen'
    }

    fig = go.Figure()

    # Add traces for each gender
    for gender in genders:
        gender_data = avg_posts[avg_posts['Gender'] == gender]
        fig.add_trace(
            go.Bar(
                x=gender_data['Platform'],
                y=gender_data['Posts_Per_Day'],
                name=gender,
                marker_color=gender_colors.get(gender, 'gray'),
                text=gender_data['Posts_Per_Day'].round(2),
                textposition='outside',
                textfont=dict(color='black', size=12, family='Arial Black'),
                visible=(gender == 'Male')  # Only 'Male' visible by default
            )
        )

    # Create dropdown menu buttons
    buttons = []
    for i, gender in enumerate(genders):
        visibility = [False] * len(genders)
        visibility[i] = True
        buttons.append(
            dict(
                label=gender,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": {
                        'text': f"<b>Average Daily Posts by Platform ({gender})</b>",
                        'font': {'size': 24},
                        'x': 0.5,
                        'xanchor': 'center'
                    }}
                ]
            )
        )

    # Final layout configuration
    fig.update_layout(
        title={
            'text': "<b>Average Daily Posts by Platform (Male)</b>",
            'font': {'size': 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title="Platform",
            tickfont=dict(family='Arial', size=14, color='black')
        ),
        yaxis=dict(
            title="Avg Posts Per Day",
            tickfont=dict(family='Arial', size=12, color='black')
        ),
        width=500,
        height=500,
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        plot_bgcolor='white'
    )

    return fig

def create_figure_4(df):
    # Count users per platform
    platform_counts = df['Platform'].value_counts().reset_index()
    platform_counts.columns = ['Platform', 'Number of Users']

    # Create figure with two bar orientations
    fig = go.Figure()

    # Column chart (vertical bars)
    fig.add_trace(
        go.Bar(
            x=platform_counts['Platform'],
            y=platform_counts['Number of Users'],
            name='Column Plot',
            visible=True
        )
    )

    # Horizontal bar chart
    fig.add_trace(
        go.Bar(
            x=platform_counts['Number of Users'],
            y=platform_counts['Platform'],
            orientation='h',
            name='Bar Plot',
            visible=False
        )
    )

    # Dropdown toggle
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(
                        label="Column Plot",
                        method="update",
                        args=[
                            {"visible": [True, False]},
                            {
                                "xaxis": {"title": "Platform"},
                                "yaxis": {"title": "Number of Users"}
                            }
                        ]
                    ),
                    dict(
                        label="Bar Plot",
                        method="update",
                        args=[
                            {"visible": [False, True]},
                            {
                                "xaxis": {"title": "Number of Users"},
                                "yaxis": {"title": "Platform"}
                            }
                        ]
                    )
                ],
                direction="down",
                x=0.5,
                xanchor="center",
                y=1.15,
                yanchor="top"
            )
        ],
        title={
            'text': "<b>Most Popular Platforms</b>",
            'font': {'size': 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Platform",
        yaxis_title="Number of Users",
        plot_bgcolor='white',
        height=400,
        width=800
    )

    return fig


def create_figure_5(df):
    # Prepare data
    likes = df.groupby('Dominant_Emotion')['Likes_Received_Per_Day'].mean()
    comments = df.groupby('Dominant_Emotion')['Comments_Received_Per_Day'].mean()
    emotions = likes.index.tolist()

    # Define traces
    trace_likes = go.Bar(
        x=emotions,
        y=likes,
        name='Likes',
        marker_color='#66c2a5',
        visible=True
    )

    trace_comments = go.Bar(
        x=emotions,
        y=comments,
        name='Comments',
        marker_color='#fc8d62',
        visible=False
    )

    trace_both_likes = go.Bar(
        x=emotions,
        y=likes,
        name='Likes',
        marker_color='#66c2a5',
        visible=False
    )

    trace_both_comments = go.Bar(
        x=emotions,
        y=comments,
        name='Comments',
        marker_color='#fc8d62',
        visible=False
    )

    # Create figure
    fig = go.Figure(data=[trace_likes, trace_comments, trace_both_likes, trace_both_comments])

    # Define dropdown buttons
    buttons = [
        dict(
            label='Likes',
            method='update',
            args=[{'visible': [True, False, False, False]},
                  {'title': 'Average Likes Received per Emotion',
                   'barmode': 'group',
                   'yaxis': {'title': 'Average Likes'},
                   'xaxis': {'title': 'Dominant Emotion'}}]
        ),
        dict(
            label='Comments',
            method='update',
            args=[{'visible': [False, True, False, False]},
                  {'title': 'Average Comments Received per Emotion',
                   'barmode': 'group',
                   'yaxis': {'title': 'Average Comments'},
                   'xaxis': {'title': 'Dominant Emotion'}}]
        ),
        dict(
            label='Both (Stacked)',
            method='update',
            args=[{'visible': [False, False, True, True]},
                  {'title': 'Average Engagement per Emotion (Stacked)',
                   'barmode': 'stack',
                   'yaxis': {'title': 'Average Engagement'},
                   'xaxis': {'title': 'Dominant Emotion'}}]
        ),
    ]

    # Layout configuration
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.5,
                xanchor='center',
                y=1.15,
                yanchor='top'
            )
        ],
        template='plotly_white',
        height=400,
        width=1200,
        title={
            'text': "<b>Average Likes Received per Emotion</b>",
            'font': {'size': 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Dominant Emotion',
        yaxis_title='Average Likes'
    )

    return fig

def create_figure_6(df):
    # Create Age_Group column with bins and labels
    age_bins = [21, 24, 29, 35]
    age_labels = ['21-24', '25-29', '30-35']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)

    # Group by Age_Group and calculate average daily usage time
    age_grouped = df.groupby('Age_Group')['Daily_Usage_Time (minutes)'].mean().reset_index()

    # Create bar plot
    fig = px.bar(
        age_grouped,
        x='Age_Group',
        y='Daily_Usage_Time (minutes)',
        title="Average Daily Time Spent on Social Media by Age Group",
        labels={
            'Daily_Usage_Time (minutes)': 'Average Daily Usage Time (minutes)',
            'Age_Group': 'Age Group'
        },
        color='Age_Group',
        color_discrete_sequence=px.colors.sequential.Mint
    )

    # Update layout
    fig.update_layout(
        height=400,
        width=600,
        plot_bgcolor='white',
        title={
            'text': "<b>Average Social Media Use by Age</b>",
            'font': {'size': 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=False
    )

    return fig

def create_age_groups(age):
    if 21 <= age <= 24:
        return '21-24'
    elif 25 <= age <= 29:
        return '25-29'
    elif 30 <= age <= 35:
        return '30-35'
    else:
        return 'Other'

df['Age_Group'] = df['Age'].apply(create_age_groups)


# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.MINTY])
server = app.server  

age_group_items = [
    dbc.DropdownMenuItem("All Ages", id="age-all", n_clicks=1),
    dbc.DropdownMenuItem("From 21 to 24", id="age-21-24", n_clicks=0),
    dbc.DropdownMenuItem("From 25 to 29", id="age-25-29", n_clicks=0),
    dbc.DropdownMenuItem("From 30 to 35", id="age-30-35", n_clicks=0),
]


navbar = dbc.Navbar(
    html.Div(
        [
            html.Div(style={"flex": "1"}),  
            dbc.NavbarBrand(
                html.Span(
                    "Emotion Analysis Dashboard",
                    style={'fontWeight': 'bold', 'fontSize': '50px', 'color': 'white'}
                ),
                href="#",
                style={"margin": "0"}
            ),
            html.Div([
                dcc.Dropdown(
                    id='age-dropdown',
                    options=[
                        {'label': 'All Age Groups', 'value': 'All'},
                        {'label': '21-24 years', 'value': '21-24'},
                        {'label': '25-29 years', 'value': '25-29'},
                        {'label': '30-35 years', 'value': '30-35'}
                    ],
                    value='All',
                    placeholder="Select Age Group",
                    style={'width': '180px', 'display': 'inline-block'}
                )],
                style={"flex": "1", "display": "flex", "justifyContent": "flex-end"}
            ),  
        ],
        style={
            'width': '100%',
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'paddingLeft': '20px',
            'paddingRight': '20px',
            'margin': '0 auto'
        }
    ),
    color="primary",
    dark=True,
    sticky="top",
    style={
        "width": "100%", 
        "maxWidth": "2680px",  
        "margin": "0 auto",  
        "padding": "0"
    }
)

buttons = html.Div([
    dbc.Button(
        "Home Page",
        id="home-page-btn",
        color="primary",
        size="lg",
        style={
            "marginRight": "20px",
            "fontWeight": "bold",
            "fontSize": "22px",
            "padding": "12px 30px"
        }
    ),
    dbc.Button(
        "Test Your Emotion",
        id="test-emotion-btn",
        color="secondary",
        size="lg",
        style={
            "fontWeight": "bold",
            "fontSize": "22px",
            "padding": "12px 30px"
        }
    )
], style={
    'display': 'flex',
    'justifyContent': 'center',
    'alignItems': 'center',
    'marginTop': '30px',
    'marginBottom': '30px'
})

def get_home_content(fig1, fig2, fig3, fig4, fig5, fig6):
    return html.Div([
        html.Div([
            html.Div(dcc.Graph(id='emotions-by-usage-time', figure=fig2), 
                    style={'padding': '5px', 'border': '2px solid #ccc', 'borderRadius': '2px','boxShadow': '0 4px 8px rgba(0, 0, 0, 0.15)'}),
            html.Div(dcc.Graph(id='top-emotions-platform', figure=fig1), 
                    style={'padding': '5px', 'border': '2px solid #ccc', 'borderRadius': '2px','boxShadow': '0 4px 8px rgba(0, 0, 0, 0.15)'}),
            html.Div(dcc.Graph(id='avg-posts-platform-gender', figure=fig3), 
                    style={'padding': '5px', 'border': '2px solid #ccc', 'borderRadius': '2px','boxShadow': '0 4px 8px rgba(0, 0, 0, 0.15)'}),
            html.Div(dcc.Graph(id='platform-popularity-toggle', figure=fig4), 
                    style={'padding': '5px', 'border': '2px solid #ccc', 'borderRadius': '2px','boxShadow': '0 4px 8px rgba(0, 0, 0, 0.15)'}),
            html.Div(dcc.Graph(id='emotion-engagement-plot', figure=fig5), 
                    style={'padding': '5px', 'border': '2px solid #ccc', 'borderRadius': '2px','boxShadow': '0 4px 8px rgba(0, 0, 0, 0.15)'}),
            html.Div(dcc.Graph(id='daily-usage-by-age', figure=fig6), 
                    style={'padding': '5px', 'border': '2px solid #ccc', 'borderRadius': '2px','boxShadow': '0 4px 8px rgba(0, 0, 0, 0.15)'}),
        ], style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(3, 1fr)',
            'gridGap': '10px',
            'padding': '0 20px',
            'justifyContent': 'center'
        })
    ])

# Test emotion page content
def get_test_emotion_content():
    return html.Div([
        html.H2(
            "🎯 Predict Your Dominant Emotion Based on Social Media Usage",
            style={
                'textAlign': 'center',
                'color': '#3AAFA9',
                'marginBottom': '40px',
                'fontWeight': 'bold'
            }
        ),

        html.Div([
            # Left Column
            html.Div([
                html.Label("Age", style={'color': '#2B7A78', 'fontWeight': '600'}),
                dcc.Input(id='age', type='number', min=10, max=100,
                          style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                 'border': '1px solid #95E1D3', 'marginBottom': '15px'}),

                html.Label("Gender", style={'color': '#2B7A78', 'fontWeight': '600'}),
                dcc.Dropdown(id='gender', options=[
                    {'label': 'Male', 'value': 'Male'},
                    {'label': 'Female', 'value': 'Female'}
                ], style={'borderRadius': '5px', 'marginBottom': '15px'}),

                html.Label("Mostly Used Platform", style={'color': '#2B7A78', 'fontWeight': '600'}),
                dcc.Dropdown(id='platform', options=[
                    {'label': 'Instagram', 'value': 'Instagram'},
                    {'label': 'Facebook', 'value': 'Facebook'},
                    {'label': 'Twitter', 'value': 'Twitter'},
                    {'label': 'TikTok', 'value': 'TikTok'},
                    {'label': 'YouTube', 'value': 'YouTube'},
                    {'label': 'LinkedIn', 'value': 'LinkedIn'},
                    {'label': 'Snapchat', 'value': 'Snapchat'}
                ], style={'borderRadius': '5px', 'marginBottom': '15px'}),

                html.Label("Daily Usage Time (minutes)", style={'color': '#2B7A78', 'fontWeight': '600'}),
                dcc.Input(id='daily', type='number',
                          style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                 'border': '1px solid #95E1D3', 'marginBottom': '15px'}),
            ], style={
                'width': '48%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'paddingRight': '20px',
                'boxSizing': 'border-box'
            }),

            # Right Column
            html.Div([
                html.Label("Posts Per Day", style={'color': '#2B7A78', 'fontWeight': '600'}),
                dcc.Input(id='posts', type='number',
                          style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                 'border': '1px solid #95E1D3', 'marginBottom': '15px'}),

                html.Label("Likes Received Per Day", style={'color': '#2B7A78', 'fontWeight': '600'}),
                dcc.Input(id='likes', type='number',
                          style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                 'border': '1px solid #95E1D3', 'marginBottom': '15px'}),

                html.Label("Comments Received Per Day", style={'color': '#2B7A78', 'fontWeight': '600'}),
                dcc.Input(id='comments', type='number',
                          style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                 'border': '1px solid #95E1D3', 'marginBottom': '15px'}),

                html.Label("Messages Sent Per Day", style={'color': '#2B7A78', 'fontWeight': '600'}),
                dcc.Input(id='messages', type='number',
                          style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                 'border': '1px solid #95E1D3', 'marginBottom': '15px'}),
            ], style={
                'width': '48%',
                'display': 'inline-block',
                'verticalAlign': 'top'
            }),
        ], style={
            'maxWidth': '1100px',
            'margin': 'auto',
            'backgroundColor': '#E6F7F6',
            'padding': '30px',
            'borderRadius': '12px',
            'boxShadow': '0 4px 12px rgba(0,0,0,0.1)',
            'marginBottom': '40px'
        }),

        html.Div([
            html.Button("Predict Emotion", id='predict-button', n_clicks=0,
                        style={
                            'backgroundColor': '#3AAFA9',
                            'color': 'white',
                            'border': 'none',
                            'padding': '14px 34px',
                            'borderRadius': '25px',
                            'cursor': 'pointer',
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'display': 'block',
                            'margin': '0 auto'
                        }),
        ], style={'marginBottom': '40px'}),

        html.Div(id='prediction-output',
                 style={'fontSize': 24, 'textAlign': 'center', 'color': '#17252A', 'fontWeight': 'bold', 'minHeight': '40px'}),

        html.Div(id='emotion-image', style={'textAlign': 'center', 'marginTop': '20px'}),

        html.Div(id='insight-plot',
                 style={'maxWidth': '1100px', 'margin': 'auto', 'marginTop': '30px'})
    ], style={
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#F0F9F8',
        'padding': '50px 30px'
    })


# Main app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content'),
    buttons
], style={'maxWidth': '2680px', 'margin': 'auto'})

# Callback to update page content based on URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('age-dropdown', 'value')]

)
def display_page(pathname, selected_age):
    if pathname == '/test-emotion':
        return get_test_emotion_content()
    else:  # Default to home page
        if selected_age is None or selected_age == 'All':
            filtered_df = df
        else:
            filtered_df = df[df['Age_Group'] == selected_age]
        if filtered_df.empty:
            return html.Div([
                html.H3(f"No data available for selected age group {selected_age}", 
                       style={'textAlign': 'center', 'color': '#17252A'})
            ])
        
        # Recreate figures with filtered data
        filtered_fig1 = create_figure_1(filtered_df)
        filtered_fig2 = create_figure_2(filtered_df)
        filtered_fig3 = create_figure_3(filtered_df)
        filtered_fig4 = create_figure_4(filtered_df)
        filtered_fig5 = create_figure_5(filtered_df)
        filtered_fig6 = create_figure_6(filtered_df)
        return get_home_content(filtered_fig1, filtered_fig2, filtered_fig3, 
                               filtered_fig4, filtered_fig5, filtered_fig6)

# Callback for Home Page button
@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    Input('home-page-btn', 'n_clicks'),
    prevent_initial_call=True
)
def go_to_home(n_clicks):
    if n_clicks:
        return '/'

# Callback for Test Your Emotion button
@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    Input('test-emotion-btn', 'n_clicks'),
    prevent_initial_call=True
)
def go_to_test_emotion(n_clicks):
    if n_clicks:
        return '/test-emotion'
    
@app.callback(
    Output('prediction-output', 'children'),
    Output('insight-plot', 'children'),
    Output('emotion-image', 'children'), 
    Input('predict-button', 'n_clicks'),
    State('age', 'value'),
    State('gender', 'value'),
    State('platform', 'value'),
    State('daily', 'value'),
    State('posts', 'value'),
    State('likes', 'value'),
    State('comments', 'value'),
    State('messages', 'value'),
    prevent_initial_call=True
)
def predict_emotion(n_clicks, age, gender, platform, daily, posts, likes, comments, messages):
    if n_clicks == 0:
        return "", "", ""

    # Validate inputs
    if any(v is None for v in [age, gender, platform, daily, posts, likes, comments, messages]):
        return "Please fill in all fields", "", ""

    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Platform': platform,
        'Daily_Usage_Time (minutes)': daily,
        'Posts_Per_Day': posts,
        'Likes_Received_Per_Day': likes,
        'Comments_Received_Per_Day': comments,
        'Messages_Sent_Per_Day': messages
    }])

    prediction = model.predict(input_data)[0]
    print(prediction)
    
    avg_values = df[df['Dominant_Emotion'] == prediction][[
        'Daily_Usage_Time (minutes)', 'Posts_Per_Day', 'Likes_Received_Per_Day',
        'Comments_Received_Per_Day', 'Messages_Sent_Per_Day'
    ]].mean().reset_index()

    avg_values.columns = ['Feature', 'Average']
    avg_values['User Value'] = [daily, posts, likes, comments, messages]

    fig = px.bar(avg_values, x='Feature', y=['Average', 'User Value'],
                 barmode='group',
                 title=f"Your Usage vs Average for {prediction}",
                 color_discrete_sequence=['#2B7A78', '#99D98C'])

    fig.update_layout(
        plot_bgcolor='#E6F7F6',
        paper_bgcolor='#E6F7F6',
        font_color='#17252A'
    )

    # Image handling
    image_filename = f"{prediction}.png"
    
    try:
        # Check if image file exists
        full_image_path = os.path.join('assets', image_filename)
        if os.path.exists(full_image_path):
            img_component = html.Img(src=f"/assets/{prediction}.png", style={"height": "100px"})
        else:
            # Fallback if image doesn't exist
            img_component = html.Div([
                html.P(f"🎭 {prediction}", 
                       style={'fontSize': '48px', 'margin': '0'}),
                html.P("(Image not found)", 
                       style={'fontSize': '12px', 'color': '#888'})
            ])
    except Exception as e:
        # Fallback for any other errors
        img_component = html.Div([
            html.P(f"🎭 {prediction}", 
                   style={'fontSize': '48px', 'margin': '0'})
        ])

    return f"🧠 Predicted Dominant Emotion: {prediction}", dcc.Graph(figure=fig), img_component

# Run app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8050)
