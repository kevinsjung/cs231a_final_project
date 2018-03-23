import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random
from sklearn import metrics

images_file = 'output3.csv'
scores_file = 'validation_scores.csv'

images_df = pd.read_csv(images_file)
scores_df = pd.read_csv(scores_file)

images_df['ImageName'] = [x.lower() for x in images_df['ImageName'].values]
scores_df['subDirectory_filePath'] = [x.lower() for x in scores_df['subDirectory_filePath'].values]

images_df = images_df.set_index('ImageName')
scores_df = scores_df.set_index(['subDirectory_filePath'])

print scores_df.index.values
print images_df.index.values


# generate X and Y scores
images_all = images_df.index.values

print images_all

valence_X_all = images_df['LipRatio'].values              # lip ratio
arousal_X_all = images_df['Gap'].values                   # mouth gap

images = []
valence_X = []
valence_Y = []                                        # validation valence score
arousal_X = []
arousal_Y = []                                        # validation arousal score

for i in range(len(images_all)):
    image = images_all[i]
    try:
        image_scores = scores_df.loc[image]
    except:
        print "image not found: ", image
    if image_scores['valence_category'] != -1 and image_scores['arousal_category'] != -1:
        valence_X.append(valence_X_all[i])
        arousal_X.append(arousal_X_all[i])
        valence_Y.append(image_scores['valence_category'])
        arousal_Y.append(image_scores['arousal_category'])


plt.plot(valence_X, valence_Y, linewidth='0.08')
plt.savefig("valence_dist.pdf", bbox_inches='tight', pad_inches=0.2)
plt.title('Valence Distribution')
plt.show()


plt.plot(arousal_X, arousal_Y, linewidth='0.08')
plt.savefig("arousal_dist.pdf", bbox_inches='tight', pad_inches=0.2)
plt.title('Arousal Distribution')
plt.show()


## VALENCE ##

# divide into training and testing sets (n-fold cross-validation)
numSamples = len(valence_X)

# downsample for valence
valence_Y0_ind = [i for i in range(numSamples) if valence_Y[i] == 0]
valence_Y1_ind = [i for i in range(numSamples) if valence_Y[i] == 1]

valence_Y1_ind_downsample = np.random.choice(valence_Y1_ind, len(valence_Y0_ind), replace=False)

valence_X_downsample = np.array([valence_X[i] for i in range(numSamples) if i in valence_Y0_ind or i in valence_Y1_ind_downsample])
valence_Y_downsample = np.array([valence_Y[i] for i in range(numSamples) if i in valence_Y0_ind or i in valence_Y1_ind_downsample])

numSamples = len(valence_X_downsample)

downsample_random_ind = np.random.choice(range(numSamples),numSamples,replace=False)
valence_X_downsample = valence_X_downsample[downsample_random_ind]
valence_Y_downsample = valence_Y_downsample[downsample_random_ind]

print len(valence_X_downsample)
print len(valence_Y_downsample)


folds = 10
numSamples = len(valence_X_downsample)
numTestSamples = numSamples/folds + 1
numTrainSamples = numSamples - numTestSamples
valence_accuracies = []
valence_nulls = []
valence_confusion_matrix = np.zeros([2,2])

for i in range(0, numSamples, numTestSamples):
    j = i+numTestSamples
    if j > numSamples:
        j = numSamples
    test_ind = range(i,j)
    
    valence_train_X = np.array([valence_X_downsample[i] for i in range(numSamples) if i not in test_ind])
    valence_test_X = np.array([valence_X_downsample[i] for i in test_ind])
    valence_train_Y = np.array([valence_Y_downsample[i] for i in range(numSamples) if i not in test_ind])
    valence_test_Y = np.array([valence_Y_downsample[i] for i in test_ind])
    
    valence_train_X_downsample = valence_train_X_downsample.reshape(-1, 1)
    valence_test_X = valence_test_X.reshape(-1,1)
    valence_train_Y_downsample = valence_train_Y_downsample.reshape(-1, 1)
    
#     print valence_train_X_downsample.shape[0]
#     print valence_train_Y_downsample.shape[0]
#     print valence_test_X.shape[0]
#     print valence_test_Y.shape[0]

    valenceModel = LogisticRegression()
    valenceModel.fit(valence_train_X_downsample, valence_train_Y_downsample)
    accuracy = valenceModel.score(valence_test_X, valence_test_Y)
    predicted = valenceModel.predict(valence_test_X)

    nullaccuracy = sum(valence_test_Y) / len(valence_test_Y)
    
    valence_accuracies.append(accuracy)
    valence_nulls.append(nullaccuracy)

    confmat = metrics.confusion_matrix(valence_test_Y, predicted)
    valence_confusion_matrix = valence_confusion_matrix + confmat


## AROUSAL ##

# divide into training and testing sets (n-fold cross-validation)
numSamples = len(arousal_X)

# downsample for arousal
arousal_Y0_ind = [i for i in range(numSamples) if arousal_Y[i] == 0]
arousal_Y1_ind = [i for i in range(numSamples) if arousal_Y[i] == 1]

arousal_Y1_ind_downsample = np.random.choice(arousal_Y1_ind, len(arousal_Y0_ind), replace=False)

arousal_X_downsample = np.array([arousal_X[i] for i in range(numSamples) if i in arousal_Y0_ind or i in arousal_Y1_ind_downsample])
arousal_Y_downsample = np.array([arousal_Y[i] for i in range(numSamples) if i in arousal_Y0_ind or i in arousal_Y1_ind_downsample]) 

numSamples = len(arousal_X_downsample)

downsample_random_ind = np.random.choice(range(numSamples),numSamples,replace=False)
arousal_X_downsample = arousal_X_downsample[downsample_random_ind]
arousal_Y_downsample = arousal_Y_downsample[downsample_random_ind]

print len(arousal_X_downsample)
print len(arousal_Y_downsample)

folds = 10
numSamples = len(arousal_X_downsample)
numTestSamples = numSamples/folds + 1
numTrainSamples = numSamples - numTestSamples
arousal_accuracies = []
arousal_nulls = []
arousal_confusion_matrix = np.zeros([2,2])


for i in range(0, numSamples, numTestSamples):
    j = i+numTestSamples
    if j > numSamples:
        j = numSamples
    test_ind = range(i,j)
    
    arousal_train_X = np.array([arousal_X_downsample[i] for i in range(numSamples) if i not in test_ind])
    arousal_test_X = np.array([arousal_X_downsample[i] for i in test_ind])
    arousal_train_Y = np.array([arousal_Y_downsample[i] for i in range(numSamples) if i not in test_ind])
    arousal_test_Y = np.array([arousal_Y_downsample[i] for i in test_ind])
    
    arousal_train_X_downsample = arousal_train_X_downsample.reshape(-1, 1)
    arousal_test_X = arousal_test_X.reshape(-1,1)
    arousal_train_Y_downsample = arousal_train_Y_downsample.reshape(-1, 1)
    
#     print arousal_train_X_downsample.shape[0]
#     print arousal_train_Y_downsample.shape[0]
#     print arousal_test_X.shape[0]
#     print arousal_test_Y.shape[0]

    arousalModel = LogisticRegression()
    arousalModel.fit(arousal_train_X_downsample, arousal_train_Y_downsample)
    accuracy = arousalModel.score(arousal_test_X, arousal_test_Y)
    predicted = arousalModel.predict(arousal_test_X)

    nullaccuracy = sum(arousal_test_Y) / len(arousal_test_Y)
    
    arousal_accuracies.append(accuracy)
    arousal_nulls.append(nullaccuracy)
    
    confmat = metrics.confusion_matrix(arousal_test_Y, predicted)
    arousal_confusion_matrix = arousal_confusion_matrix + confmat


## PRINT RESULTS ## 

print np.mean(valence_accuracies)
print np.mean(valence_nulls)
print valence_confusion_matrix
print '\n'
print np.mean(arousal_accuracies)
print np.mean(arousal_nulls)
print arousal_confusion_matrix


## PLOT RESULTS ##

plt.plot(valence_accuracies, marker='', color='olive', linewidth=2, label="model accuracies")
plt.plot(valence_nulls, marker='', color='olive', linewidth=2, linestyle='dashed', label="null accuracies")
plt.legend()
#plt.title('Valence')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.savefig("valence.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()



plt.plot(arousal_accuracies, marker='', color='olive', linewidth=2, label="model accuracies")
plt.plot(arousal_nulls, marker='', color='olive', linewidth=2, linestyle='dashed', label="null accuracies")
plt.legend()
#plt.title('Arousal')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.savefig("arousal.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()

