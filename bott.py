from sklearn import tree

# weight, 1 for smooth -- 0 for rough
features = [[140, 1], [130,1], [150,0], [170,0]]
# 0 for apple, 1 for orange
labels = [0, 0, 1, 1]

# Makes desicion tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# weight, 1 for smooth -- 0 for rough
print(clf.predict ([[200, 0]]))
