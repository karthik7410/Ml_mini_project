import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
import base64
import random
import os

# Function to add a styled background image
def add_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)), url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            color: white;
        }}
        h1 {{
            font-size: 64px !important;
            text-align: center;
            margin-top: 100px;
        }}
        p {{
            font-size: 28px !important;
            text-align: center;
        }}
        .stButton > button {{
            display: block;
            margin: 40px auto;
            padding: 20px 50px;
            font-size: 28px;
            background-color: #ff4b4b !important;
            color: white !important;
            border-radius: 12px;
            border: none;
            font-weight: bold;
        }}
        .logout-button button {{
            background-color: #ff4b4b !important;
            color: white !important;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 20px;
            float: right;
            margin-top: 10px;
            margin-right: 20px;
        }}
        /* Dark Mode */
        .stApp {{
            background-color: #2e2e2e;
            color: white;
        }}
        .stButton > button {{
            background-color: #ff4b4b !important;
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Load and preprocess the data
@st.cache_data
def load_data():
    ratings_url = "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat"
    movies_url = "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat"
    
    ratings = pd.read_csv(ratings_url, sep="::", engine='python', names=["userId", "movieId", "rating", "timestamp"])
    movies = pd.read_csv(movies_url, sep="::", engine='python', names=["movieId", "title", "genres"])
    
    # Handle genres
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    movies = movies[~movies['genres'].apply(lambda x: 'Adult' in x)].reset_index(drop=True)

    # One-hot encode genres
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies['genres'])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

    # Merge with original
    movies = pd.concat([movies[['movieId', 'title']], genre_df], axis=1)

    return movies, genre_df.columns.tolist(), ratings

# Recommend K random genre-similar movies
def recommend_movies_by_genre(genre, movies_df, ratings_df, k=5):
    genre_movies = movies_df[movies_df[genre] == 1].reset_index(drop=True)

    if genre_movies.empty or len(genre_movies) < k:
        return []

    random_indices = random.sample(range(len(genre_movies)), k)
    features = genre_movies.drop(columns=['movieId', 'title'])

    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(features)

    query_index = random.choice(random_indices)
    distances, indices = model.kneighbors([features.iloc[query_index]], n_neighbors=k + 1)

    recommendations = []
    for i in range(1, k + 1):
        movie_row = genre_movies.iloc[indices[0][i]]
        movie_id = movie_row['movieId']
        movie_title = movie_row['title']
        avg_rating = ratings_df[ratings_df['movieId'] == movie_id]['rating'].mean()
        recommendations.append((movie_title, round(avg_rating, 2) if not pd.isna(avg_rating) else "No rating"))

    return recommendations

# ------------------ STREAMLIT APP ------------------

st.set_page_config(page_title="Movie Recommender", layout="wide")

# Session state
if "visited" not in st.session_state:
    st.session_state.visited = False

# Landing Page
if not st.session_state.visited:
    bg_image_path = os.path.join(os.getcwd(), "Movie.jpeg")  # Relative path to the image
    add_bg_from_local(bg_image_path)

    st.markdown("<h1>Welcome to the Movie Recommender App!</h1>", unsafe_allow_html=True)
    st.markdown("<p>Select a genre and get amazing movie recommendations instantly!</p>", unsafe_allow_html=True)

    if st.button("🎬 Enter the App"):
        st.session_state.visited = True
        st.rerun()

# Main App
if st.session_state.visited:
    bg_image_path = os.path.join(os.getcwd(), "Movie.jpeg")  # Relative path to the image
    add_bg_from_local(bg_image_path)

    # Logout Button (styled and right-aligned)
    logout_button = st.button("🚪 Logout", key="logout_button", help="Click to logout and go back to the landing page")

    if logout_button:
        st.session_state.visited = False
        st.rerun()

    st.markdown("<h1 style='text-align: center;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;'>Get <strong>random</strong> movie recommendations based on your favorite genre using <strong>KNN + cosine similarity</strong>!</p>", unsafe_allow_html=True)

    # Load data
    movies_df, genre_list, ratings_df = load_data()

    # Genre dropdown
    selected_genre = st.selectbox("Choose a genre:", genre_list)

    if st.button("Recommend Movies"):
        recommendations = recommend_movies_by_genre(selected_genre, movies_df, ratings_df, k=5)

        if recommendations:
            # Styled Recommendations Table
            table_html = """
                <style>
                    .recommendation-table {{
                        width: 90%;
                        margin: auto;
                        background-color: rgba(255, 255, 255, 0.9);
                        border-radius: 15px;
                        padding: 20px;
                        color: black;
                        font-size: 18px;
                        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
                    }}
                    .recommendation-table table {{
                        width: 100%;
                        border-collapse: collapse;
                    }}
                    .recommendation-table th, .recommendation-table td {{
                        text-align: left;
                        padding: 12px;
                        border-bottom: 1px solid #ccc;
                    }}
                    .recommendation-table th {{
                        background-color: #ff4b4b;
                        color: white;
                    }}
                    .recommendation-table td:last-child {{
                        width: 15%;
                        text-align: center;
                    }}
                </style>
                <div class="recommendation-table">
                    <h3>Top Recommendations for <em>{genre}</em> genre</h3>
                    <table>
                        <tr>
                            <th>Movie Title</th>
                            <th>Rating ⭐</th>
                        </tr>
                        {rows}
                    </table>
                </div>
            """

            rows_html = ""
            for movie, rating in recommendations:
                rows_html += f"<tr><td>{movie}</td><td>{rating}</td></tr>"

            final_html = table_html.format(genre=selected_genre, rows=rows_html)
            st.markdown(final_html, unsafe_allow_html=True)
        else:
            st.warning("No recommendations found for the selected genre.")
