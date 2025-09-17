import pandas as pd
import pickle as pk
import numpy as np
import re
import string
import spacy

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('./db/datas/train.csv', sep=',')
test = pd.read_csv('./db/datas/test.csv', sep=',')

IDs = test['id']
test_data = test.drop('id', axis=1)
Y_data = data['target']
X_data = data.drop(['id', 'target'], axis=1)


nlp = spacy.load('en_core_web_lg')

object_cols = X_data.select_dtypes(include='object').columns.to_list()
urls_pattern = re.compile(r'https?://\S+|www\.\S+|\S+\.com/\S+|\S+\.org/\S+|t\.co/\S+')
pontuaciton_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
nunbers_pattern = re.compile(r'\d+')


datas = [X_data, test_data]
for i, data in enumerate(datas):
    data = data.replace('', np.nan)
    data = data.replace('-', np.nan)
    data['keyword'] = data['keyword'].fillna('unknown_keyword')
    data['location'] = data['location'].fillna('unknown_location')

    for col in object_cols:
        data[col] = data[col].astype('string')
        data[col] = data[col].apply(lambda s: s.lower())
        data[col] = data[col].apply(lambda text: urls_pattern.sub(r'', text).strip())
        data[col] = data[col].apply(lambda text: pontuaciton_pattern.sub(r'', text).strip())
        data[col] = data[col].apply(lambda text: re.sub(r'\s+', ' ', text).strip())

    data['location'] = data['location'].astype('category')
    data['keyword'] = data['keyword'].astype('category')

    data['text'] = data['text'].apply(lambda text: nunbers_pattern.sub(r'', text).strip())
    data['text'] = data['text'].apply(lambda text: " ".join([token.lemma_ for token in nlp(text) if not token.is_stop]))
    datas[i] = data

data_train, data_test = datas

categoric_cols = data_train.select_dtypes(include='category').columns.to_list()

O_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
vectorization = TfidfVectorizer(min_df=5, max_df=0.85, max_features=7500, ngram_range=(1, 2))

X_train_categoric_cols = O_encoder.fit_transform(data_train[categoric_cols])
X_train_string_cols = vectorization.fit_transform(data_train['text'])
features_names = O_encoder.get_feature_names_out(categoric_cols)
feature_vector = vectorization.get_feature_names_out()

X_test_categoric_cols = O_encoder.transform(data_test[categoric_cols])
X_test_string_cols = vectorization.transform(data_test['text'])

X_train_categoric_df = pd.DataFrame(X_train_categoric_cols, columns=features_names, index=data_train.index)
X_train_string_df = pd.DataFrame(X_train_string_cols.toarray(), columns=feature_vector, index=data_train.index)
X_test_categoric_df = pd.DataFrame(X_test_categoric_cols, columns=features_names, index=data_test.index)
X_test_string_df = pd.DataFrame(X_test_string_cols.toarray(), columns=feature_vector, index=data_test.index)

X_train_processed = pd.concat([X_train_categoric_df, X_train_string_df], axis=1)
X_test_processed = pd.concat([X_test_categoric_df, X_test_string_df], axis=1)

with open('db/processed/data_pre_processed.pkl', mode='wb') as f:
    pk.dump([X_train_processed, Y_data], f)

with open('./db/submission/test.pkl', mode='wb') as f:
    pk.dump(X_test_processed, f)

IDs.to_csv('./db/submission/IDs.csv', sep=',', index=False)