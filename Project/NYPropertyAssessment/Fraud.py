import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# Read data and create 45 expert variables
df = pd.read_csv('UnfinishedCleaningData.csv')


def CreateExpertVariable(df):
    df_new = pd.DataFrame()
    variables = list(df.columns)[5:]
    groups = list(df.columns)[1:5]
    df['All'] = 1
    groups.append('All')

    for variable in variables:
        for group in groups:
            df_new[variable + '_By_' + group] = \
                (df[variable] / df.groupby(group)[variable].transform('mean'))

    return df_new


df2 = CreateExpertVariable(df)
df_scale_1 = scale(df2)


def PCAModel(df, n=20):
    model = PCA(n_components=n)
    model.fit(df)

    return model


def PCAPlot(model):
    print("The percent of variance explained by first", model.n_components_, "dimensions is",
          round(sum(model.explained_variance_ratio_), 4))

    features = range(model.n_components_)
    plt.bar(features, model.explained_variance_)
    plt.xticks(features)
    plt.ylabel('variance')
    plt.xlabel('PCA feature')


# Model 1 with 20 dimensions
model1 = PCAModel(df_scale_1,20)
PCAPlot(model1)


# Model 2 with 8 dimensions
model2 = PCAModel(df_scale_1,8)
PCAPlot(model2)

# Transform and scale again
df_transformed = model2.transform(df_scale_1)
df_scale_2 = scale(df_transformed)

df_output = pd.DataFrame(df_scale_2)
#df_output.to_csv("ReadyForScore.csv")


# Build Score 1 using Heuristic Function

score_1 = np.sqrt(np.square(df_output).sum(axis=1))
    
# Build Score 2 using Autoencoder

from keras.models import Model, load_model
from keras.layers import Input, Dense

#df = pd.read_csv('ReadyForScore.csv')
#df = df.iloc[:,1:]

# Build the model
encoding_dim = 4
input_layer = Input(shape=(8,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(8, activation='relu')(encoder)
autoencoder = Model(input_layer, decoder)


autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])


autoencoder.fit(df_scale_2, df_scale_2,
                epochs=50,
                batch_size=32,
                shuffle=True)

prediction = autoencoder.predict(df_scale_2)

# pred = pd.DataFrame(prediction)
# pred.to_csv('prediction.csv')

score_2 = np.sqrt(np.square(prediction - df_scale_2).sum(axis=1))

# Create Score dataframe and Score Ranking using Quantile binning
def ScoreandRanking():
    score = pd.DataFrame()
    score['BBLE'] = df['BBLE']
    socre["score1"] = score_1
    socre["score2"] = score_2
    
    score = score.sort_values('score1')
    score['ranking_1'] = np.arange(1, df_new.shape[0]+1)
    score = score.sort_values('score2')
    score['ranking_2'] = np.arange(1,df_new.shape[0]+1)
    score['combined_ranking'] = score['ranking_1'] + score['ranking_2']
    score = score.sort_values('combined_ranking',ascending=False)

    return score

ScoreandRanking()

# Score Distribution
plt.hist(score['score_1'],bins=30)
plt.xlabel('Fraud Score 1 (Heuristic Function)')
plt.ylabel('Count')
plt.title('The Distribution of Fraud Score 1 (Heuristic Function)')
plt.yscale('log');

plt.hist(score['score_2'],bins=30)
plt.xlabel('Fraud Score 2 (Autoencoder)')
plt.ylabel('Count')
plt.title('The Distribution of Fraud Score 2 (Autoencoder)')
plt.yscale('log');




