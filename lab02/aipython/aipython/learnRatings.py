# learnRatings.py - Rating data for linear learner
# AIFCA Python code Version 0.9.18 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2026 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import matplotlib.pyplot as plt
import urllib.request
import csv
from statistics import mean
from display import Displayable
from learnProblem import Data_set, Learner, Evaluate
from learnNoInputs import Naive_learner
from learnLinear import Linear_learner

class Data_set_from_ratings(Data_set):
    def __init__(self,
                 time_split=891382267, # 90% training
                 #time_split=888259984, # 75% training
                 #time_split=880845177, # 60% training
                 #time_split=882826944, # halfway split
                 ratings_file_name="ml-100k/u.data",
                 item_file_name="ml-100k/u.item",
                 genres_start=5,
                 ratings_separator = '\t',
                 movie_separator="|"):
        """
        ratings before time_split are training set. Those after are test.
        genres_start is starting position of genres in item file

        This code assumes that the older 100k dataset is downloaded from 
        https://grouplens.org/datasets/movielens/
        Unzip ml-100k.zip into the current directory.
        """
        self.genres = ["Action", "Adventure", "Animation", "Children",
                       "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                       "Film-Noir", "Horror", "Musical", "Mystery",
                       "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        genre2pos = {genre:pos for (pos,genre) in enumerate(self.genres)}
        self.users = set()  #set of users


        self.display(1,"Movie Rating Dataset.  Reading...")
        with open(ratings_file_name,'r') as ratings_file:
            #all_ratings = csv.reader(ratings_file, delimiter=ratings_separator) 
            all_ratings = (line.strip().split(ratings_separator) for line in ratings_file)

            # Get rating information:
            self.training_data = []
            self.test_data = []
            self.timestamps = []
            self.training_stats = {1:0, 2:0, 3:0, 4:0 ,5:0}
            self.test_stats = {1:0, 2:0, 3:0, 4:0 ,5:0}
            for quadruple in all_ratings:
                (user,item,rating,timestamp) = tuple(int(v) for v in quadruple)
                self.timestamps.append(timestamp) # useful for picking split
                self.users.add(user)
                if timestamp <= time_split:
                    self.training_data.append((user,item,rating))
                    self.training_stats[rating] += 1
                else:
                    self.test_data.append((user,item,rating))
                    self.test_stats[rating] += 1
            self.training_mean = (sum(r*n for (r,n) in self.training_stats.items())
                                             /sum(self.training_stats.values())) 

            # Get movie information:
        with open(item_file_name,'r', encoding='ISO-8859-1') as item_file:
            self.item_titles = {}
            self.item_props = {}
            self.extra_genres = {}
            item_lines = (line.strip().split(movie_separator) for line in item_file)
            #item_lines = csv.reader(item_file, delimiter=movie_separator)
            for items in item_lines:
                movieId = int(items[0])
                self.item_titles[movieId] = items[1]
                self.item_props[movieId] = [int(v) for v in items[genres_start:]]

            # the following is to create pseudo examples (see text)
            self.item_props[-1] = [1 for _ in self.genres] # special movie
            self.item_props[-2] = [0 for _ in self.genres]

            Data_set.__init__(self, self.training_data, test=self.test_data, prob_valid=0,
                                  target_index=2, header=self.genres, target_type="numeric")
            # Create input features:
            # input features are properties of the item (e[1]) for rating e
            self.input_features = []
            for i in range(len(self.header)):
                def feat(e, ind=i):
                    return self.item_props[e[1]][ind]
                feat.type = "boolean"
                feat.__doc__ = self.header[i]
                self.input_features.append(feat)
            self.show_stats()

    def show_stats(self):
        self.display(1, len(self.training_data),"training ratings and",
                len(self.test_data),"test ratings")
        self.tr_users = {user for (user,item,rating) in self.training_data}
        self.test_users = {user for (user,item,rating) in self.test_data}
        self.display(1,"users:",len(self.tr_users),"training,",len(self.test_users),"test,",
                     len(self.tr_users & self.test_users),"in common")
        tr_items = {item for (user,item,rating) in self.training_data}
        test_items = {item for (user,item,rating) in self.test_data}
        self.display(1,"items:",len(tr_items),"training,",len(test_items),"test,",
                     len(tr_items & test_items),"in common")
        self.display(1,"Rating statistics for training set: ",self.training_stats)
        self.display(1,"Mean training rating: ",self.training_mean)
        self.display(1,"Rating statistics for test set: ",self.test_stats)

def test():
    global movielens, naive_learner, naive_predictor, global_learner, global_predictor
    movielens = Data_set_from_ratings()
    naive_learner = Naive_learner(movielens)
    naive_predictor = naive_learner.learn()
    naive_learner.evaluate()
    global_learner =  Linear_learner(movielens, learning_rate = 0.001, squashed=False)
    global_predictor = global_learner.learn(num_iter=10000)
    global_learner.evaluate()

if __name__ == "__main__":
    test()

class UserDataSets(Displayable):
    def __init__(self, ratings):
        """ ratings is a Data_set_from_ratings
        """
        self.ratings = ratings
        
    def create_user_datatsets(self, pseudo):
        """
        create a dataset for each user.
        pseudo is the number of each of the pseudo examples added (see code document)
        """
        self.pseudo = pseudo
        self.user2train = {} # user to list of training examples dict
        self.user2test = {}  # user to list of test examples dict
        self.pseudo_egs = [(-1,-1,self.ratings.training_mean),
                               (-1,-2,self.ratings.training_mean)]* pseudo
        for (user,item,rating) in self.ratings.train:
            if user not in self.user2train:
                self.user2train[user] = self.pseudo_egs.copy()
            self.user2train[user].append((user,item,rating))
        for (user,item,rating) in self.ratings.test:
            if user in self.user2test:
                self.user2test[user].append((user,item,rating))
            else:
                self.user2test[user] = [(user,item,rating)]
        self.user2dataset = {}
        for user in self.user2test:
            if user in self.user2train:
                self.user2dataset[user] =  A_users_ratings(self.ratings, self.user2train[user], self.user2test[user])
        return self.user2dataset

    def learn_personalized(self, pseudo=0, ecrit=Evaluate.squared_loss):
        """
        learn personalized ratings for each user.
        this assumes that test() is run to create the naive_learner and global_learner
        """
        self.create_user_datatsets(pseudo=pseudo)
        self.learners = {}
        x_values = []
        pers_values = []
        naive_values = []
        global_values = []
        self.display(1,"user \t#train \t#test \tnaive \tglobal \tpersonal")
        total_num_train = 0
        total_num_test = 0
        for user in self.user2dataset:
            num_train = len(self.user2dataset[user].train)-2*self.pseudo
            total_num_train += num_train
            if num_train == 0: #no training data
                predictor = naive_predictor
                self.learners[user] = naive_learner
            else:
                lnr = Linear_learner(self.user2dataset[user], max_init=0.1,
                                     learning_rate = 0.001, squashed=False)
                predictor = lnr.learn(num_iter=1000)
                self.learners[user] = lnr
            error = self.user2dataset[user].evaluate_dataset(self.user2dataset[user].test, predictor, ecrit)
            x_values.append(num_train)
            pers_values.append(error)
            naive_error = self.user2dataset[user].evaluate_dataset(self.user2dataset[user].test,
                                                    naive_predictor, ecrit)
            naive_values.append(naive_error)
            global_error = self.user2dataset[user].evaluate_dataset(self.user2dataset[user].test,
                                                    global_predictor, ecrit)
            global_values.append(global_error)
            total_num_test += len(self.user2dataset[user].test)
            self.display(1,f"{user}\t{num_train}\t{len(self.user2dataset[user].test)}\t"
                             f"{naive_error:.5f}\t{global_error:.5f}\t{error:.5f}")
        self.display(1,f"mean\t{total_num_train/len(self.user2dataset):.1f}\t"
                         f"{total_num_test/len(self.user2dataset):.1f}\t{mean(naive_values):.5f}\t"
                         f"{mean(global_values):.5f}\t{mean(pers_values):.5f}")
        self.display(1,f"{pseudo=}")
        plt.ion()
        fig, ax = plt.subplots()
        # ax.set_xscale('log') # lets you visualize small training datasets better
        ax.set_xlabel("training size")
        ax.set_ylabel(ecrit.__doc__)
        ax.plot(x_values, pers_values, "o", label=f"personalized {pseudo=}")
        ax.plot(x_values, naive_values, "x", label="naive")
        ax.plot(x_values, global_values, "+", label="global")
        ax.legend()
        plt.show()
            
class A_users_ratings(Data_set):
    def __init__(self, ratings_dataset, training, test):
        self.max_display_level = 0
        Data_set.__init__(self, training, test=test, prob_valid=0,
                            num_properties=ratings_dataset.num_properties,
                            target_index=ratings_dataset.target_index)
        self.input_features = ratings_dataset.input_features

if __name__ == "__main__":
    userdatasets = UserDataSets(movielens)
    userdatasets.learn_personalized(pseudo=0)

