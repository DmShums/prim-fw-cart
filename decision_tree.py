import pandas as pd
import numpy as np

class Node:
    def __init__(self, X, y, gini, predicted_class):
        self.X = X
        self.y = y
        self.gini = gini
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class MyDecisionTreeClassifier:
   
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def gini(self, classes):
        '''
        A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.
        
        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem).
        '''
        num_of_class_elems = {i: 0 for i in set(classes)}
        gini_index = 1

        for ind in classes:
            num_of_class_elems[ind] += 1

        for elem in num_of_class_elems.items():
            gini_index -= (elem[1]/len(classes))**2

        return gini_index


    def split_data(self, X, y):
        # test all the possible splits in O(N*F) where N in number of samples
        # and F is number of features

        # return index and threshold value
        df = pd.DataFrame(X)
        df_cols = list(df.columns)
        try:
            df[df_cols[-1] + 1] = pd.DataFrame(y)
        except:
            df = pd.concat([df, pd.DataFrame(y)], axis=1)
        # col for y
        col_sorted = {col: [] for col in df_cols}
        # col for x
        col_sorted_df = {col: [] for col in df_cols}

        # sort by columns
        for col in df_cols:
          col_sorted[col] = df.sort_values(by=[col])[list(df.columns)[-1]].tolist()
          col_sorted_df[col] = df.sort_values(by=[col])[list(df.columns)[col]].tolist()

        # calculate gini for each split
        best_gini = 1
        feature, threshold = None, None
        for index1, elem in enumerate(col_sorted.items()):
          for index2 in range(len(elem[1])):
            l_gini = elem[1][:index2]
            r_gini = elem[1][index2:]
            m_gini = len(l_gini)/len(elem[1]) * self.gini(l_gini) + len(r_gini)/len(elem[1]) * self.gini(r_gini)

            if elem[1][index2] == elem[1][index2-1]:
              continue

            if m_gini < best_gini:
              best_gini = m_gini
              feature = index1
              threshold = (col_sorted_df[index1][index2] + col_sorted_df[index1][index2-1])/2

        return feature, threshold

    def build_tree(self, X, y, depth = 0):   
        # create a root node

        # recursively split until max depth is not exeeced
        num_samples_per_class = [np.sum(y == i) for i in range(self.classes)]
        predicted_class = np.argmax(num_samples_per_class)

        root_node = Node(X, y, self.gini(y), predicted_class)
        if depth < self.max_depth:
          ind = self.split_data(X, y)[0]
          thr = self.split_data(X, y)[1]

          if ind is not None:
            left_X = [row for row in X if row[ind] <= thr]
            left_y = [y[i] for i, val in enumerate(X) if val[ind] <= thr]

            right_X = [row for row in X if row[ind] > thr]
            right_y = [y[i] for i, val in enumerate(X) if val[ind] > thr]

            root_node.feature_index = ind
            root_node.threshold = thr
            root_node.left = self.build_tree(left_X, left_y, depth + 1)
            root_node.right = self.build_tree(right_X, right_y, depth + 1)
          return root_node
    
    def fit(self, X, y):
        # basically wrapper for build tree / train
        self.classes = len(set(X))
        self.features = len(X)
        self.tree_ = self.build_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree_
        while node.left and node.right:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def evaluate(self, X_test, y_test):
        return sum(int((self.predict(X_test) == y_test)) == True) / len(y_test)
