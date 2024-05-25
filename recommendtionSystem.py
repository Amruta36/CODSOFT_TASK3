import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

# Sample data (replace with your dataset)
data = {
    'user_id': ['User1', 'User1', 'User2', 'User2', 'User3', 'User3', 'User4', 'User4', 'User5', 'User5'],
    'item_id': ['The Shawshank Redemption', 'Inception', 'The Shawshank Redemption', 'Inception', 'Inception',
                'The Godfather', 'The Shawshank Redemption', 'The Godfather', 'Inception', 'The Godfather'],
    'rating': [5, 4, 4, 3, 5, 3, 4, 5, 4, 5]
}

# Create Surprise dataset from pandas dataframe
reader = Reader(rating_scale=(1, 5))
df = pd.DataFrame(data)
dataset = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split dataset into train and test sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Use user-based collaborative filtering
algo = KNNBasic(sim_options={'user_based': True})

# Train the algorithm on the trainset
algo.fit(trainset)

# Get top N recommendations for a user
user_id = input("Enter your UserId (Case-sensitive): ")  # ID of the user for whom recommendations are needed
n_recommendations = int(input("Enter number of recommendations you want: "))  # Number of recommendations needed

# List of items already rated by the user
user_items = df[df['user_id'] == user_id]['item_id'].tolist()

# List of items to predict (items not yet rated by the user)
items_to_predict = [item for item in df['item_id'].unique() if item not in user_items]

# Predict ratings for items not yet rated by the user
predictions = [algo.predict(user_id, item) for item in items_to_predict]

# Sort predictions by estimated rating
sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

# Get top N recommended items
recommended_items = [prediction.iid for prediction in sorted_predictions[:n_recommendations]]

# Print the recommended items
print("Recommended items for User", user_id + ":", recommended_items)
