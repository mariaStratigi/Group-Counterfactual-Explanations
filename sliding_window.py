import pandas as pd
import numpy as np
import math
import operator
import itertools
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans


# Class for initiating and keeping track of the sliding window
# Input: arr: The list of items
#        window_size: the size of the window. The size remains static throughout the experiment
class SlidingWindow:
    def __init__(self, arr, window_size):
        self.arr = arr
        self.window_size = window_size
        self.index = 0  # Keep track of the current window

# Returns the next window and set the index to point to the following window
    def get_next_window(self):
        if self.index + self.window_size <= len(self.arr):
            window = self.arr[self.index:self.index + self.window_size]
            self.index += 1  # Move to the next window
            return window
        else:
            return None  # Return None when all windows are processed


# Changes the original data by removing the items in the potential counterfactual explanation from all the interactions
# of the group members
# Input: originalData: the original data without any changes
#        groupIds: a list with the ids of the group members
#        itemIds: a list of items - the potential counterfactual explanation
# Output: newData: the altered data
def changeData(originalData, groupIds, itemIds):
    newData = originalData.drop(
        originalData[(originalData.movieId.isin(itemIds)) & originalData.userId.isin(groupIds)].index)
    return newData


# Reads all the group ids from the file
# Input: file: the path to the file that contains the group ids
# Output: groups: a list with all the group ids
def readGroups(file):
    f = open(file, "r")
    groups = []
    for x in f:
        groups.append(x)
    return groups


# Parses through the group id and returns a list with the group members ids
# Input: group: the group id
# Output: membersIds: a list of the group members ids
def getGroupMembers(group):
    group = group.strip()
    members = group.split('_')
    membersIds = []
    for m in members:
        membersIds.append(int(m))
    return membersIds


# Returns a list of items that at least one group members has interacted with
# Input: group: a list of the group members ids
#        originalData: the original data of the system
# Output: movies: a list of distinct movie ids with which at least one group member has an interaction
def getRatedItemsByAllGroupmembers(group, originalData):
    movies = originalData[originalData.userId.isin(group)]['movieId'].unique()
    #print(movies)
    return movies


# Returns all the movie ids that noone in the group has interacted with. This is used by the recommender system
# to only make predictions for new items, i.e., movies that at least one member has rated will not be suggested to the
# group
# Input: orginalData: the original data of the system
#        movie_ids: a list of all the movie ids
#        group: a list of the group members ids
# Output: movie_ids_to_pred: a list of movie ids that none of the group members has interacted with
def getMoviesForRecommendation(originalData, movie_ids, group):
    # Get a list of all movie IDs that have been watched by the group
    movie_ids_group = originalData.loc[originalData.userId.isin(group), "movieId"]
    # Get a list off all movie IDS that that have NOT been watched by group
    movie_ids_to_pred = np.setdiff1d(movie_ids, movie_ids_group)
    return movie_ids_to_pred


# Utilizing the model that is already trained, generate predictions for movies that none of the group members has any
# interaction with. For a specific group member add their individual recommendation list into a dictionary.
# Input: model: the recommendation model that is already trained
#        user_id_str: the group member id
#        movie_ids_to_pred: a list of movie ids that none of the group members has interacted with
# Output: predictions: a dictionary containing the movied id and its corresponding prediction for the group member
#           Key: movie id
#           Value: model's prediction for that movie and group member
def generate_recommendation(model, user_id_str, movie_ids_to_pred):
    # Apply a rating of 4 to all interactions (only to match the Surprise dataset format)
    user_id = int(user_id_str)
    test_set = [[user_id, movie_id, 4] for movie_id in movie_ids_to_pred]
    # Predict the ratings and generate recommendations

    predictions = model.test(test_set)
    pred_ratings = np.array([pred.est for pred in predictions])
    index_max = (-pred_ratings).argsort()[:]
    predictions = {}
    for i in index_max:
        predictions[movie_ids_to_pred[i]] = pred_ratings[i]
    return predictions


# The group recommender: for the group recommender model, we aggregate the prediction scores from all group members for
# an item into one group score. The aggregation function we use is the average.
# Input: predictions: a dictionary that contains the groups members and their predictions
#           Key: the user id
#           Value: a dictionary containing the predictions for that user
#                   Key: movie ids
#                    Value: the predicted score
#        groupSize: the size of the group
#        flag: an indicator that determines if the function returns the top-1 or the top-flag
#               flag = 0, return top-1
#               flag > 0, return top-flag
#               flag < 0, return the entire list
#               flag > number of distinct movies in predictions, return the entire list
def groupRecommendations(predictions, groupSize, flag):
    scores = {}
    for user, pred in predictions.items():
        for m in pred:
            if m in scores:
                scores[m] = scores[m] + pred[m]
            else:
                scores[m] = pred[m]
    groupPred = {}
    for m in scores:
        groupPred[m] = scores[m] / groupSize
    sorted_pred = dict(sorted(groupPred.items(), key=operator.itemgetter(1), reverse=True))
    # If flag = 0  return only the top 1
    # else return the top-k, where k = flag
    if flag == 0:
        res = next(iter(sorted_pred))
    else:
        listRec = []
        i = 0
        for key in sorted_pred:
            listRec.append(key)
            i = i + 1
            if i == flag:
                break
        res = listRec
    return res


# Calculates the average item intensity for a counterfactual explanation. Used in the experiments.
# Input: e: the counterfactual explanation
#        group: a list containg the group members ids
#        data: the original data in the system
# Output: explanationIntensity: the average item intensity of the counterfactual explanation
def findAverageItemIntensityExplanation(e, group, data):
    explanationIntensity = []
    groupint64 = []
    for g in group:
        gg = np.int64(g)
        groupint64.append(gg)
    for item in e:
        tmp = []
        tmp.append(item)
        intensity = len(data[(data.movieId.isin(tmp) & data.userId.isin(groupint64))])
        intensity = intensity / len(group)
        explanationIntensity.append(intensity)
    #
    return explanationIntensity


# Calculates the user intensity of an explanation. Used in the experiments.
# Input: e: a counterfactual explanation
#        group: a list of the group members ids
#        data: the original data of the system
# Output: userIntensity: the average user intensity of the explanation
def findUserIntensity(e, group, data):
    userIntensity = []

    for mm in group:
        m = np.int64(mm)
        tmp = []
        tmp.append(m)
        intensity = len(data[(data.movieId.isin(e) & (data.userId.isin(tmp)))])
        intensity = intensity / len(e)
        userIntensity.append(intensity)
    return userIntensity


# Finds the popularity of each movie (i.e., how many ratings each movie has received), and normalize that value to
# range [0,1]
# Input: movies: a list of all movie ids
#        data: the original data of the system
# Output: popMask: a dictionary containing the normalized popularity of the movies
#                   Key: movie id
#                   Value: normalized popularity
def popularityMask(movies, data):
    pop = []
    i = 0
    popMask = {}
    for m in movies:
        num = len(data[data["movieId"] == m])
        pop.append(num)
    # find the minimum value and range, and add 1% padding
    range_value = max(pop) - min(pop)
    range_value = range_value + range_value / 50
    min_value = min(pop) - range_value / 100

    # subtract the minimum value and divide by the range
    i = 0
    for m in movies:
        popMask[m] = ((pop[i] - min_value) / range_value)
        i = i + 1
    return popMask


# Creates a dictionary that maps each group member and the prediction score that the individual recommender system
# generated for the target item. The target item is the item for which we want to produce the explanation.
# Input: target: the target item
#        predictions: a dictionary that contains the groups members and their predictions
#            Key: the user id
#            Value: a dictionary containing the predictions for that user
#                     Key: movie ids
#                     Value: the predicted score
# Output: rel: a dictionary mapping the user id and their corresponding predicted score for the target item
#                   Key: user id
#                   Value: the user's predicted score for the target item
def relevanceMask(target, predictions):
    rel = {}
    for user, pred in predictions.items():
        if target in pred:
            rel[user] = pred[target]
            # print(rel[user])
        else:
            rel[user] = 0
    return rel


# For each item calculates the average prediction score that was generated by the individual recommender system
# for the group members. It then normalizes that sccore to the range [0,1].
# Input: m: the movie id
#        data: the original data of the system
#        relMask: the relevance Mask generated from function 'relevanceMask'
#        group: a list with the group members ids
# Output: score: the average score
def findRelevance(m, data, relMask, group):
    score = 0
    i = 0
    for gm in group:
        dd = data[(data["userId"] == gm) & (data["movieId"] == m)]
        if dd.empty:
            continue
        score = score + relMask[gm]
        i = i + 1

    if i == 0:
        return 0
    score = score / i
    maxV = 5
    minV = 0
    range_value = maxV - minV
    range_value = range_value + range_value / 50
    min_value = minV - range_value / 100
    score = (score - min_value) / range_value
    return score


# Calculates the item intensity. Finds for a movie how many in the group have rated it. Then normalize it by the
# size of the group
# Input: m: the movie id
#        group: a list of the group members ids
#        data: the original data of the system
# Output: num: the average item intensity
def findItemIntensity(m, group, data):
    num = len(data[(data.movieId == m) & data.userId.isin(group)])
    num = num / len(group)
    return num


# Calculates the average ratings given to the movie by the group members. Then normalizes that value to [0,1]
# Input: m: the movie id
#        data: the original data of the system
#        group: a list of the group members ids
# Output: score: the average ratings given to the movie by the group members
def findRatings(m, data, group):
    df = data[(data.movieId == m) & data.userId.isin(group)]
    score = df["rating"].sum()
    score = score / len(group)
    maxV = 5
    minV = 0
    range_value = maxV - minV
    range_value = range_value + range_value / 50
    min_value = minV - range_value / 100
    score = (score - min_value) / range_value
    return score


# For each item calculates a score based on the items: popularity, average predicted score, average relevance to the
# group and item intensity. Then adds these items to a dictionary and sorts it in descending order based on the scores.
# Input: itemRatedByGroup: a list of items that at least one group member has interacted with.
#        data: the original data of the system
#        members: a list of the group members ids
#        relMask: the relevance mask
#        popMask: the popularity mask
# Output: lst: the sorted list of movie ids. Sorted based on the accumulative scores of popularity, item intensity,
#              ratings and relevance
def makeChart(itemRatedByGroup, data, members, relMask, popMask):
    chart = {}
    for i in itemRatedByGroup:
        pop = popMask[i]
        itemIntensity = findItemIntensity(i, members, data)
        avgRatings = findRatings(i, data, members)
        rel = findRelevance(i, data, relMask, members)

        item_score = pop + rel + itemIntensity + avgRatings
        chart[i] = item_score

    sorted_exp = dict(sorted(chart.items(), key=operator.itemgetter(1), reverse=True))
    lst = list(sorted_exp.keys())
    return lst


# Start of the experiments
# Read the ratings file.
reader = Reader(rating_scale=(1, 5))
cols = list(pd.read_csv("C:/Users/jomast/Documents/Datasets/MovieLens/ml-latest-small/ratings.csv", nrows=1))
movielens = pd.read_csv("C:/Users/jomast/Documents/Datasets/MovieLens/ml-latest-small/ratings.csv", sep=",", header=0,
                        usecols=[i for i in cols if i != "timestamp"])

movie_ids = movielens["movieId"].unique()

sim_options = {
    "name": "pearson",
    "user_based": True,
    "min_support": 5,
    "verbose ": False,
}
# Train the recommendation model
data = Dataset.load_from_df(movielens[["userId", "movieId", "rating"]], reader)
algo = KNNWithMeans(sim_options=sim_options, verbose=False)
trainingSet = data.build_full_trainset()
algo.fit(trainingSet)

# Read the file with the group ids
all_groups = readGroups('C:/Users/jomast/Documents/Datasets/MovieLens/groups/groupsWithModerateRatings5untested.txt')


for group in all_groups:
    group = group.strip('\n')
    members = getGroupMembers(group)
    print(members)
    candidateMovies = getMoviesForRecommendation(movielens, movie_ids, members)

    # Get the recommendations for all group members
    predictions = {}
    for m in members:
        user_pred = generate_recommendation(algo, m, candidateMovies)
        predictions[m] = user_pred
    # get the group recommendation. originalGroupRec: the target item -> item we want a counterfactual explanation for
    originalGroupRec = groupRecommendations(predictions, 5, 0)
    found = 0
    # get all the items that at least one group member has interacted with
    rawitemRatedByGroup = getRatedItemsByAllGroupmembers(members, movielens)

    itemRatedByGroup = rawitemRatedByGroup
    s = len(itemRatedByGroup)

    calls = 0

    explanationsFound = {}
    # Create the popularity and Relevance Mask
    popMask = popularityMask(itemRatedByGroup, movielens)
    relMask = relevanceMask(originalGroupRec, predictions)

    # Create the aggregated list based on the aggregation of the four metrics
    chart = makeChart(itemRatedByGroup, movielens, members, relMask, popMask)
    #print("------------------")

    # NOTE: change the following between static size window and percentage window
    #window_size = math.floor(len(chart) * 0.1)
    window_size = 10

    print("window size: {}\tchart size: {}".format(window_size,len(chart)))
    # Create the window
    sw = SlidingWindow(chart, window_size)

    checked = []
    s = len(chart)
    if not chart:
        print("Could not find any items")
        continue

    l = 1
    exp = []
    wind_count = 0
    while True:
        # Get the sliding window
        big_window = sw.get_next_window()
        # If the window has passed through all list exit with message: Could not find Explanation
        # If the explanation is found exit
        # If the group recommended system has been called more than 1000 times stop
        if big_window is None:
            print("Could not find Explanation")
            break
        if found > 0:
            break
        if calls > 1000:
            break

        # counts how many times the group recommender has been called
        calls = calls + 1
        wind_count = wind_count + 1

        # remove from the group interactions the items in the sliding window
        changedData = changeData(movielens, members, big_window)

        # Retrain the recommendation model
        data1 = Dataset.load_from_df(changedData[["userId", "movieId", "rating"]], reader)
        algo1 = KNNWithMeans(sim_options=sim_options, verbose=False)
        trainingSet1 = data1.build_full_trainset()
        algo1.fit(trainingSet1)

        # Get the group recommendation
        predictions1 = {}
        for m in members:
            user_pred = generate_recommendation(algo1, m, candidateMovies)
            predictions1[m] = user_pred
        groupRec = groupRecommendations(predictions1, 5, 10)
        # if the target item is still in the group recommendation list continue
        if originalGroupRec in groupRec:
            continue
        else:
            # a counterfactual explanation has been found
            found = found + 1
            length = 1
            found_subset = 0
            # For the items in the sliding window find all possible combinations
            for length in range(len(big_window)):
                length = length + 1
                if found_subset > 0:
                    break
                combinations = itertools.combinations(big_window, length)

                for it in combinations:
                    # if a counterfactual explanation is found stop
                    # if the group recommender system has been called more than 1000 stop
                    if found_subset > 0:
                        break
                    else:
                        if calls > 1000:
                            break
                    exp = []

                    for itm in range(length):
                        exp.append(it[itm])

                    calls = calls + 1
                    # print('{} of {} at combination length {}'.format(l,cs,length))

                    # print(calls)
                    # print('checking set: {}'.format(exp))

                    # Change the data and call on the group recommender system
                    changedData = changeData(movielens, members, exp)

                    data1 = Dataset.load_from_df(changedData[["userId", "movieId", "rating"]], reader)
                    algo1 = KNNWithMeans(sim_options=sim_options, verbose=False)
                    trainingSet1 = data1.build_full_trainset()
                    algo1.fit(trainingSet1)

                    predictions1 = {}
                    for m in members:
                        user_pred = generate_recommendation(algo1, m, candidateMovies)
                        predictions1[m] = user_pred
                    groupRec = groupRecommendations(predictions1, 5, 10)
                    if originalGroupRec in groupRec:
                        l = l + 1
                        continue
                    else:
                        found_subset = found_subset + 1
                        print('Explanation: {} : found at call: {}'.format(exp, calls))
                        itemIntensity = findAverageItemIntensityExplanation(exp, members, movielens)
                        userIntensity = findUserIntensity(exp, members, movielens)
                        explanationsFound[calls] = exp
                        exp_size = len(exp)
                        print('{}\t{}\t{}\t{}'.format(exp_size, calls, itemIntensity, userIntensity))
                    l = l + 1


            l = l + 1

    if found == 0:
        print("Explanation could not be found")
print('done')
