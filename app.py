import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import joblib
import base64
import os

# Load data and model
df = pd.read_csv("data/social_media_emotions.csv")
model = joblib.load("rf_pipeline.pkl")

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # For Render

# Navbar
navbar = html.Div([
    html.Div([
        html.H2("Emotion Dashboard", style={"margin": "0", "color": "white"}),
        html.Div([
            html.Label("Filter by Age:", style={"color": "white", "margin-right": "10px"}),
            dcc.Dropdown(
                id="age-filter",
                options=[{"label": str(age), "value": age} for age in sorted(df["age"].unique())],
                multi=True,
                placeholder="Select age(s)"
            )
        ], style={"width": "300px"})
    ], style={"display": "flex", "justify-content": "space-between", "align-items": "center",
              "padding": "10px", "background-color": "#1a1a1a"})
])

# Home Page
home_layout = html.Div([
    navbar,
    html.Br(),
    html.Div([
        dcc.Graph(id="bar-plot", style={"width": "48%", "display": "inline-block"}),
        dcc.Graph(id="pie-chart", style={"width": "48%", "display": "inline-block"})
    ]),
    html.Div([
        dcc.Graph(id="line-chart", style={"width": "48%", "display": "inline-block"}),
        dcc.Graph(id="scatter-plot", style={"width": "48%", "display": "inline-block"})
    ]),
    html.Div([
        dcc.Graph(id="heatmap", style={"width": "48%", "display": "inline-block"}),
        dcc.Graph(id="parallel-coordinates", style={"width": "48%", "display": "inline-block"})
    ])
])

# Emotion Prediction Page
test_emotion_layout = html.Div([
    navbar,
    html.Br(),
    html.Div([
        html.Label("Enter Daily Time Spent on Social Media (mins):"),
        dcc.Input(id="input-daily", type="number", placeholder="e.g., 120"),

        html.Label("Enter Number of Posts Per Week:"),
        dcc.Input(id="input-posts", type="number", placeholder="e.g., 10"),

        html.Label("Enter Number of Likes Per Day:"),
        dcc.Input(id="input-likes", type="number", placeholder="e.g., 50"),

        html.Label("Enter Age:"),
        dcc.Input(id="input-age", type="number", placeholder="e.g., 25"),

        html.Label("Enter Platform (Instagram, Twitter, Facebook, TikTok):"),
        dcc.Input(id="input-platform", type="text", placeholder="e.g., Instagram"),

        html.Button("Predict Emotion", id="predict-button")
    ], style={"display": "flex", "flexDirection": "column", "width": "50%"}),

    html.Br(),
    html.Div(id="prediction-output"),
    html.Div(id="emotion-image"),
    html.Div(id="insight-plot")
])

# App Layout
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# Page Navigation
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/test_emotion":
        return test_emotion_layout
    return home_layout

# Update Graphs Based on Age Filter
@app.callback(
    [Output("bar-plot", "figure"),
     Output("pie-chart", "figure"),
     Output("line-chart", "figure"),
     Output("scatter-plot", "figure"),
     Output("heatmap", "figure"),
     Output("parallel-coordinates", "figure")],
    [Input("age-filter", "value")]
)
def update_graphs(selected_ages):
    filtered_df = df[df["age"].isin(selected_ages)] if selected_ages else df

    fig1 = px.bar(filtered_df["emotion"].value_counts().reset_index(),
                  x="index", y="emotion", labels={"index": "Emotion", "emotion": "Count"},
                  title="Emotion Distribution")

    fig2 = px.pie(filtered_df, names="emotion", title="Emotion Proportions")

    fig3 = px.line(filtered_df.groupby("age")["likes"].mean().reset_index(),
                   x="age", y="likes", title="Average Likes by Age")

    fig4 = px.scatter(filtered_df, x="posts_per_week", y="likes", color="emotion",
                      title="Posts vs Likes Colored by Emotion")

    corr_matrix = filtered_df[["daily_time_spent", "posts_per_week", "likes", "age"]].corr()
    fig5 = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")

    fig6 = px.parallel_coordinates(filtered_df, dimensions=["daily_time_spent", "posts_per_week", "likes", "age"],
                                   color=filtered_df["emotion"].astype("category").cat.codes,
                                   labels={"daily_time_spent": "Daily Time", "posts_per_week": "Posts",
                                           "likes": "Likes", "age": "Age"},
                                   title="Parallel Coordinates of User Behavior")

    return fig1, fig2, fig3, fig4, fig5, fig6

# Emotion Prediction Callback
@app.callback(
    [Output("prediction-output", "children"),
     Output("emotion-image", "children"),
     Output("insight-plot", "children")],
    [Input("predict-button", "n_clicks")],
    [Input("input-daily", "value"),
     Input("input-posts", "value"),
     Input("input-likes", "value"),
     Input("input-age", "value"),
     Input("input-platform", "value")]
)
def predict_emotion(n_clicks, daily, posts, likes, age, platform):
    if n_clicks is None:
        return "", "", ""
    if None in [daily, posts, likes, age, platform]:
        return "Please fill in all fields", "", ""

    input_df = pd.DataFrame([{
        "daily_time_spent": daily,
        "posts_per_week": posts,
        "likes": likes,
        "age": age,
        "platform": platform
    }])

    prediction = model.predict(input_df)[0]

    # Encode and show emotion image
    image_path = f"assets/{prediction}.png"
    if os.path.exists(image_path):
        encoded_image = base64.b64encode(open(image_path, "rb").read()).decode()
        image_component = html.Img(src="data:image/png;base64,{}".format(encoded_image),
                                   style={"width": "200px", "height": "200px"})
    else:
        image_component = html.Div("Image not found.")

    # Compare user to average
    avg_metrics = df[["daily_time_spent", "posts_per_week", "likes", "age"]].mean()
    fig = px.bar(
        x=["Daily Time", "Posts/Week", "Likes", "Age"] * 2,
        y=[daily, posts, likes, age] + list(avg_metrics),
        color=["User"] * 4 + ["Average"] * 4,
        barmode="group",
        title="Your Metrics vs. Average"
    )

    return f"Predicted Emotion: {prediction}", image_component, dcc.Graph(figure=fig)

# Run app
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050)
