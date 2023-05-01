import numpy as np

def cosine_sim(query_vec, food_docs, results, ratings, restaurant_to_index, rest_docs, rest_vec):
  scores = food_docs.dot(query_vec.T).flatten()
  for i, score in enumerate(scores):
    restaurant = results[i]['restaurant']
    # currently weighing ratings very heavily since it is on larger scale, will
    # adjust weights soon
    results[i]['overall_score'] = score + ratings.get(restaurant) + 0 if rest_vec \
    is None else rest_docs[restaurant_to_index[restaurant]].dot(rest_vec)
  results.sort(key=lambda x : -x['overall_score'])
  return results[:10]




