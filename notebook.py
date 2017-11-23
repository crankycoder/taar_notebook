"""
This is a port of the ensemble notebook
"""

import csv

from collections import defaultdict
import json
import operator as op
import os
import os.path

from random import sample

import numpy as np
from scipy.spatial.distance import hamming, canberra
from scipy.optimize import minimize

from taar.recommenders import CollaborativeRecommender
from taar.recommenders import LegacyRecommender
from taar.recommenders import LocaleRecommender
from taar.recommenders.similarity_recommender import SimilarityRecommender

TOP_ADDONS_BY_LOCALE_FILE_PATH = "top_addons_by_locale.json"

COLUMNS = ['bookmark_count', 'client_id', 'disabled_addons_ids',
           'geo_city', 'installed_addons', 'locale', 'os',
           'profile_age_in_weeks', 'profile_date',
           'submission_age_in_weeks', 'submission_date',
           'subsession_length', 'tab_open_count', 'total_uri',
           'unique_tlds']


def parser(fname):
    reader = csv.reader(open(fname, 'r'))
    for row in reader:
        yield dict(zip(COLUMNS, row))


def load_rowdata():
    rows = []
    for fname in os.listdir('data'):
        for rdata in parser(os.path.join('data', fname)):
            rows.append(rdata)
    return rows


def java_string_hashcode(s):
    h = 0
    for c in s:
        h = (31 * h + ord(c)) & 0xFFFFFFFF
    return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def positive_hash(s):
    return java_string_hashcode(s) & 0x7FFFFF


class NewCollaborativeRecommender(CollaborativeRecommender):
    def recommend(self, client_data, limit):
        recommendations = self.get_weighted_recommendations(client_data)

        # Sort the suggested addons by their score and return the
        # sorted list of addon ids.
        sorted_dists = sorted(recommendations.items(),
                              key=op.itemgetter(1), reverse=True)
        return [s[0] for s in sorted_dists[:limit]]

    def get_weighted_recommendations(self, client_data):
        # Addons identifiers are stored as positive hash values within
        # the model.
        installed_addons =\
            [positive_hash(addon_id) for addon_id in
             client_data.get('installed_addons', [])]

        # Build the query vector by setting the position of the
        # queried addons to 1.0 and the other to 0.0.
        query_vector = np.array([1.0 if (entry.get("id") in installed_addons)
                                 else 0.0 for entry in self.raw_item_matrix])

        # Build the user factors matrix.
        user_factors = np.matmul(query_vector, self.model)
        user_factors_transposed = np.transpose(user_factors)

        # Compute the distance between the user and all the addons in
        # the latent space.
        distances = {}
        for addon in self.raw_item_matrix:
            # We don't really need to show the items we requested.
            # They will always end up with the greatest score. Also
            # filter out legacy addons from the suggestions.
            hashed_id = str(addon.get("id"))
            if (hashed_id in installed_addons or
                    hashed_id not in self.addon_mapping or
                    self.addon_mapping[hashed_id].get("isWebextension", False)
                    is False):
                continue

            dist = np.dot(user_factors_transposed, addon.get('features'))
            # Read the addon ids from the "addon_mapping" looking it
            # up by 'id' (which is an hashed value).
            addon_id = self.addon_mapping[hashed_id].get("id")
            distances[addon_id] = dist

        return defaultdict(int, distances)


def cdist(dist, A, b):
    return np.array([dist(a, b) for a in A])


CATEGORICAL_FEATURES = ["geo_city", "locale", "os"]
CONTINUOUS_FEATURES = ["subsession_length", "bookmark_count",
                       "tab_open_count", "total_uri", "unique_tlds"]


class NewSimilarityRecommender(SimilarityRecommender):
    def get_similar_donors(self, client_data):
        """
        Computes a set of :float: similarity scores between a client
        and a set of candidate donors for which comparable variables
        have been measured.

        A custom similarity metric is defined in this function that
        combines the Hamming distance for categorical variables with
        the Canberra distance for continuous variables into a
        univariate similarity metric between the client and a set of
        candidate donors loaded during init.

        :param client_data: a client data payload including a subset
                            of telemetry fields.
        :return: the sorted approximate likelihood ratio (np.array)
                 corresponding to the internally computed similarity score and
                 a list of indices that link each LR score with the related
                 donor in the |self.donors_pool|.
        """
        client_categorical_feats = [client_data.get(specified_key)
                                    for specified_key in CATEGORICAL_FEATURES]
        client_continuous_feats = [client_data.get(specified_key)
                                   for specified_key in CONTINUOUS_FEATURES]

        # Compute the distances between the user and the cached continuous
        # and categorical features.
        cont_features = cdist(canberra, self.continuous_features,
                              client_continuous_feats)

        # The lambda trick is needed to prevent |cdist| from
        # force-casting the string features to double.
        cat_features = cdist(hamming, self.categorical_features,
                             client_categorical_feats)

        # Take the product of similarities to attain a univariate
        # similarity score.  Addition of 0.001 to the continuous
        # features avoids a zero value from the categorical variables,
        # allowing categorical features precedence.
        distances = (cont_features + 0.001) * cat_features

        # Compute the LR based on precomputed distributions that
        # relate the score to a probability of providing good addon
        # recommendations.

        lrs_from_scores = \
            np.array([self.get_lr(distances[i])
                      for i in range(self.num_donors)])

        # Sort the LR values (descending) and return the sorted values
        # together with the original indices.
        indices = (-lrs_from_scores).argsort()
        return lrs_from_scores[indices], indices

    def get_weighted_recommendations(self, client_data):
        recommendations = defaultdict(int)

        for donor_score, donor in zip(*self.get_similar_donors(client_data)):
            for addon in self.donors_pool[donor]['active_addons']:
                recommendations[addon] += donor_score

        return recommendations


class NewLocaleRecommender(LocaleRecommender):
    def __init__(self, TOP_ADDONS_BY_LOCALE_FILE_PATH):
        LocaleRecommender.__init__(self)

        with open(TOP_ADDONS_BY_LOCALE_FILE_PATH) as data_file:
            top_addons_by_locale = json.load(data_file)

        self.top_addons_by_locale = defaultdict(lambda: defaultdict(int),
                                                top_addons_by_locale)

    def get_weighted_recommendations(self, client_data):
        client_locale = client_data.get('locale', None)
        return defaultdict(int, self.top_addons_by_locale[client_locale])


class NewLegacyRecommender(LegacyRecommender):
    def get_weighted_recommendations(self, client_data):
        recommendations = defaultdict(int)
        addons = client_data.get('disabled_addons_ids', [])

        for addon in addons:
            for replacement in self.legacy_replacements.get(addon, []):
                recommendations[replacement] += 1

        return recommendations


row_data = load_rowdata()


def client_filter(client):
    return len(client['installed_addons']) >= 1


useful_clients = filter(client_filter, row_data)
training, test = useful_clients.randomSplit([0.8, 0.2])


def random_partition(A, k):
    n = len(A)
    A = list(A)
    indices = set(sample(range(n), k))

    first = []
    second = []

    for i in range(n):
        element = A[i]

        if i in indices:
            first.append(element)
        else:
            second.append(element)

    return first, second


def get_num_masked(addons):
    return max(1, len(addons) / 2)


def mask_addons(client):
    addons = client['installed_addons']
    num_mask = get_num_masked(addons)

    masked, unmasked = random_partition(addons, num_mask)

    client['installed_addons'] = unmasked
    client['masked_addons'] = masked

    return client


training_masked = map(mask_addons, training)

recommenders = {
    "collaborative": CollaborativeRecommender(),
    "similarity": SimilarityRecommender(),
    "locale": LocaleRecommender("./top_addons_by_locale.json"),
    "legacy": LegacyRecommender()
}

def compute_features(client_data):
    recommendations = []
    matrix = []

    for _, recommender in recommenders.items():
        recommendations.append(recommender.get_weighted_recommendations(client_data))

    for addon in whitelist:
        matrix.append([features[addon] for features in recommendations])

    return client_data, np.array(matrix)


X_unnormalized = map(compute_features, training_masked)


max_feature_values = X_unnormalized.map(lambda (_, features): np.max(features, axis=0)).reduce(np.maximum)


def preprocess_locale_scores(scores):
    return np.sqrt(np.sqrt(scores))


def scale_features((client, features)):
    features = features / max_feature_values
    features[:, 0] = preprocess_locale_scores(features[:, 0])
    return client, features


# In[ ]:

X = X_unnormalized.map(scale_features)


# ## Making recommendations

# Computing recommendations then reduces down to a dot product. These results are then sorted.

# In[ ]:

def get_weighted_recommendations(client_data, features, weights):
    scores = features.dot(weights)
    return client_data, np.argsort(-scores)


# ## Measuring the performance (MAP)
# 
# We use the [MAP](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html) measure as an error metric for this optimization problem. The reason for this is mostly that we only have positive data, i.e. we know addons which users like, but we don't really have a lot of data about addons that users hate.

# In[ ]:

def average_precision(client_data, recommendations):
    tp = fp = 0.
    masked = set(client_data['masked_addons'])
    precisions = []
    
    for recommendation in recommendations:
        if whitelist[recommendation] in masked:
            tp += 1
            precisions.append(tp / (tp + fp))
            if tp == len(masked):
                break
        else:
            fp += 1
    
    if len(precisions) > 0:
        return np.mean(precisions)
    else:
        return 0.


# ## Training an ensemble model

# In[ ]:



# ### Defining a cost function

# We find recommendations, compute the average precision (AP) and then calculate the mean of that (MAP). This produces a value between 0 and 1. We then subtract this value from 1 because SciPy looks for a function to minimize, not to maximize.

# In[ ]:

def cost(weights, X=X):
    weighted_recommendations = X.map(lambda (client_data, features):
                get_weighted_recommendations(client_data, features, weights)
               )
    
    AP = weighted_recommendations.map(lambda (client_data, recommendations):
             average_precision(client_data, recommendations)
            )
        
    MAP = AP.mean()
            
    return 1 - MAP


# ### Choosing an initial guess
# 
# There are many ways of choosing initial guesses. A constant vector of 1s seems to be a sensible prior (with properly normalized features it means that all recommenders are equally useful). However, randomly choosing initial values can also be useful.

# In[ ]:

def get_initial_guess_alternative(n):
    return np.ones(n)


# In[ ]:

def get_initial_guess(n):
    return np.random.random(n)


# ### Logging the optimization process
# 
# SciPy is logging the optimization process to a stdout stream that Jupyter seems to ignore. Because it's extremely useful to see how the optimization process is actually going, we define a helper function that queries `cost` and then also prints the results.

# In[ ]:

def verbose_cost(weights):
    new_cost = cost(weights)
    print "New guess:", weights, "leads to a cost of", new_cost
    return new_cost


# ### Optimizing
# 
# The 4-element vectors in the following correspond to the recommenders in this order:

# In[ ]:

recommenders.keys()


# We're using the [COBYLA](https://en.wikipedia.org/wiki/COBYLA) algorithm for optimization. There is no theoretical reason for this, it just seems to work pretty well here and finds good results fairly quickly. Of course other algorithms could be used instead of COBYLA here.

# In[ ]:

num_features = len(recommenders)
x0 = get_initial_guess(num_features)
print "Initial guess:", x0
best_weights = minimize(verbose_cost, x0, method="COBYLA", tol=1e-5).x


# ### Experimental: Grid search

# In[ ]:

from itertools import product


# In[ ]:

def grid_search(cost, parameter_space):
    for parameters in parameter_space:
        print parameters, cost(parameters)


# In[ ]:

space = product(range(3), repeat=4)
grid_search(cost, space)


# ## Comparison to old the method
# 
# To validate if the MAP numbers that we get are any good, it's useful to compare them to the results of the previous recommendation process. The following is a minimal reimplementation of this `RecommendationManager`. It's used here because we want to use our masked data instead of the data fetched from HBase.

# In[ ]:

class RecommendationManager:
    def __init__(self, recommenders):
        self.recommenders = recommenders
        
    def recommend(self, client_data, limit):
        recommendations = []
        
        for r in self.recommenders:
            recommendations += r.recommend(client_data, limit)
            
            if len(recommendations) >= limit:
                break
            
        return recommendations[:limit]


# This helper function is similar to `map(superlist.index, sublist`) but ignores elements from the sublist that don't appear in the superlist.

# In[ ]:

def list_elements_to_indices(superlist, sublist):
    result = []
    
    for a in sublist:
        for i, b in enumerate(superlist):
            if a == b:
                result.append(i)
                break
        
    return result


# In[ ]:

def evaluate_recommendation_manager(mngr, data=training_masked):
    return 1 - data        .map(lambda user: (user, mngr.recommend(user, 10)))        .map(lambda (user, recommendations): average_precision(user, list_elements_to_indices(whitelist, recommendations)))        .mean()


# As we can see, the previous recommendation manager performs much worse than our new model:

# In[ ]:

mngr = RecommendationManager([recommenders["legacy"],
                              recommenders["collaborative"], recommenders["similarity"],
                              recommenders["locale"]])
evaluate_recommendation_manager(mngr)


# However, this comparison is a little bit unfair. The locale
# recommender is generally extremely useful and can be used as a
# better baseline. With this ordering (where nearly only the locale
# recommender is queried), we get a much more comparable result. The
# results are now in the same ballpark and the ensemble is better by
# around 2%.

# In[ ]:

mngr = RecommendationManager([recommenders["locale"], 
                              recommenders["legacy"], 
                              recommenders["collaborative"], 
                              recommenders["similarity"]])

evaluate_recommendation_manager(mngr)


# In[ ]:

mngr = RecommendationManager([recommenders["locale"]])
evaluate_recommendation_manager(mngr)


# ## Test set
# 
# The results using the test set are quite similar:

# In[ ]:

test_masked = test.map(mask_addons)


# In[ ]:

X_test_unnormalized = test_masked.map(compute_features)


# In[ ]:

X_test = X_test_unnormalized.map(scale_features)


# In[ ]:

evaluate_recommendation_manager(mngr, test_masked)


# In[ ]:

cost(best_weights, X_test)


# ## Optimizing on a subset with some manual decisions
# 
# When looking at the results so far, it seems like the locale and
# collaborative recommenders are the most important ones. The legacy
# recommender is also useful, but it can rarely be used and by using
# it we don't really optimize for MAP. Changes to the similarity
# recommender's weight also only lead to a small change.
# 
# Because of this, this section first tries to optimize for
# locale/collaborative weights using a grid search. Afterwards, we fix
# these weights and find the best weight for the similarity
# recommender. This grid search is quite expensive, so we'll work on a
# smaller subset of the data.
# 
# Generally, the results using the subset are in the same ballpark,
# but of course there are still some differences.
# 
# **Disclaimer:** This section is very much experimental and the code
# is not perfectly refactored

# In[ ]:

from itertools import product


# In[ ]:

X_full = X


# In[ ]:

X = X_full.sample(False, 0.1)


# In[ ]:

n = 65
locale = np.linspace(0.2, 2.5, num=n)
collaborative = np.linspace(0.2, 2.5, num=n)


# In[ ]:

get_ipython().magic(u'time z = [cost([x, 1., y, .11], X) for x, y in product(locale, collaborative)]')


# In[ ]:

xx, yy = np.meshgrid(locale, collaborative)
x = xx.ravel()
y = yy.ravel()


# This plot isn't really informative, because there are some areas with very bad parameters which lead to the plot being mostly blue.

# In[ ]:

hb = plt.hexbin(x, y, C=z, gridsize=30, cmap='jet')
plt.colorbar(hb)
plt.xlabel("Collaborative")
plt.ylabel("Locale")
plt.show()


# To fix this, we can cut off all the results with cost > 0.65, which creates this much more interpretable result:

# In[ ]:

z2 = np.minimum(z, 0.65)


# In[ ]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:

hb = plt.hexbin(x, y, C=z2, gridsize=30, cmap='jet')
plt.colorbar(hb)
plt.xlabel("Collaborative")
plt.ylabel("Locale")
plt.show()


# The optimum seems to be around (1.3, 1.8).

# In[ ]:

list(product(locale, collaborative))[1995]


# Next, we can try to find the best argument for the similarity
# recommender when fixing the other weights.

# In[ ]:

x = np.linspace(0, 0.2, num=100)


# In[ ]:

y = [cost([1.278125, 1., 1.8171874999999997, xi], X) for xi in x]


# In[ ]:

plt.plot(x, y)


# In[ ]:

x[np.argmin(y)]


# When evaluating on all of the data, this doesn't really improve the
# recommender though.

# In[ ]:

cost([1.278125, 1., 1.8171874999999997, 0.10707070707070707], X_test)


# In[ ]:





