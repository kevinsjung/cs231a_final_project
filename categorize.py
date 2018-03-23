import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

scores_file = 'training.csv'

scores_df = pd.read_csv(scores_file)

# plt.plot(np.sort(scores_df['valence'].values))
# plt.show()

scores_df = scores_df[['subDirectory_filePath', 'valence', 'arousal']]

# non-positive (no matter arousal): -1
# neutral (0): 0 <= V < 0.05
# positive (1): V >= 0.05
# low arousal (0): A <= 0
# high arousal (1): A > 0

valence = scores_df['valence'].values
arousal = scores_df['arousal'].values
valence_category = [0]*len(scores_df)
arousal_category = [0]*len(scores_df)

for i in range(len(scores_df)):
    if valence[i] < 0:
        valence_category[i] = -1
        arousal_category[i] = -1
    else:
        if valence[i] < 0.01:
            valence_category[i] = 0
        else:
            valence_category[i] = 1
        if arousal[i] <= 0:
            arousal_category[i] = 0
        else:
            arousal_category[i] = 1

scores_df['valence_category'] = valence_category
scores_df['arousal_category'] = arousal_category

print len([i for i in valence_category if i == 0])
print len(valence_category)

scores_df.to_csv('validation_scores.csv')







