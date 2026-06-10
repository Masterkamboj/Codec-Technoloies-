import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Movie Dataset
movies = pd.DataFrame({
    'title': [
        'Toy Story',
        'Jumanji',
        'Grumpier Old Men',
        'Waiting to Exhale',
        'Father of the Bride Part II',
        'Heat',
        'Sabrina',
        'Tom and Huck',
        'Sudden Death',
        'GoldenEye'
    ],
    'genres': [
        'Animation Comedy Family',
        'Adventure Fantasy Family',
        'Comedy Romance',
        'Comedy Drama',
        'Comedy',
        'Action Crime Thriller',
        'Comedy Romance',
        'Adventure Children',
        'Action Thriller',
        'Action Adventure Thriller'
    ]
})

# Convert genres into numerical vectors
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['genres'])

# Calculate similarity
similarity = cosine_similarity(genre_matrix)

def recommend(movie_name):
    movie_name = movie_name.strip()

    if movie_name not in movies['title'].values:
        print("\nMovie not found!")
        print("\nAvailable movies:")
        for movie in movies['title']:
            print("-", movie)
        return

    index = movies[movies['title'] == movie_name].index[0]

    distances = list(enumerate(similarity[index]))

    recommended_movies = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    print(f"\nRecommended movies for '{movie_name}':\n")

    for movie in recommended_movies:
        print(movies.iloc[movie[0]]['title'])

# Main Program
print("=== Movie Recommendation System ===")
print("\nAvailable Movies:")

for movie in movies['title']:
    print("-", movie)

movie_name = input("\nEnter a movie name: ")

recommend(movie_name)