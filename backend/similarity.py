import numpy as np

def cosine_sim(query_vec, food_docs, results, ratings, restaurant_to_index, rest_docs, rest_vec):
  scores = food_docs.dot(query_vec.T).flatten()
  scores = (scores - np.amin(scores)) / (np.amax(scores) - np.amin(scores))
  top_30 = np.argsort(-scores)[:30]
  top_items = [results[i] for i in top_30]
  if rest_vec is not None:
    rest_scores = rest_docs.dot(rest_vec)
    rest_scores = (rest_scores - np.amin(rest_scores))/(np.amax(rest_scores) - np.amin(rest_scores))
  weights = np.zeros(30)
  for i, result in enumerate(top_items):
    restaurant = result['restaurant']
    top_items[i]['food_sim'] = scores[top_30[i]]
    top_items[i]['rating'] = ratings.get(restaurant)
    top_items[i]['rest_sim'] = 0 if rest_vec is None else rest_scores[restaurant_to_index[restaurant]]
    weights[i] = top_items[i]['rating'] + top_items[i]['rest_sim']
  weights = [weight/sum(weights) for weight in weights]
  top_10 = np.random.choice(top_items, size=10, replace=False, p=weights)
  # sorted_items = np.argsort([-x['overall_score'] for x in top_10])
  return list(top_10)
  # [top_10[i] for i in sorted_items]




