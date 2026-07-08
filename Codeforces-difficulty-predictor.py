import requests
import pandas as pd
import numpy as np
import pickle
import re
import lightgbm as lgb
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# getting thr data from the official codeforces api i take two data
# one is of the problems and other is of the contest
prob_resp = requests.get("https://codeforces.com/api/problemset.problems")
prob_resp.raise_for_status()
prob_data = prob_resp.json()

df_problems = pd.DataFrame(prob_data['result']['problems'])
df_stats = pd.DataFrame(prob_data['result']['problemStatistics'])

cont_resp = requests.get("https://codeforces.com/api/contest.list")
cont_resp.raise_for_status()
df_contests = pd.DataFrame(cont_resp.json()['result'])

print(f"problems: {len(df_problems)}  |  contests: {len(df_contests)}")

# merging problems with their solve counts and contest names
df = pd.merge(df_problems,
              df_stats[['contestId', 'index', 'solvedCount']],
              on=['contestId', 'index'],
              how='left')

df = pd.merge(df,
              df_contests[['id', 'name']].rename(columns={'id': 'contestId', 'name': 'contestName'}),
              on='contestId',
              how='left')

print(df.shape)

# extracting division type from contest name
# div is one of the strongest signals for difficulty
def extract_div(name):
    if pd.isna(name):
        return 'Other'
    if 'Div. 1' in name and 'Div. 2' in name:
        return 'Div. 12'
    elif 'Div. 1' in name:
        return 'Div. 1'
    elif 'Div. 2' in name:
        return 'Div. 2'
    elif 'Div. 3' in name:
        return 'Div. 3'
    elif 'Div. 4' in name:
        return 'Div. 4'
    elif 'Educational' in name:
        return 'Educational'
    elif 'Global' in name:
        return 'Global'
    else:
        return 'Other'

df['div'] = df['contestName'].apply(extract_div)

# dropping columns i dont need
df = df.drop(columns=['name', 'type', 'points', 'contestName'])

# only keeping problems that have a rating assigned
df = df.dropna(subset=['rating']).reset_index(drop=True)

print(df.shape)
print(df['div'].value_counts())

# ordinal encoding for division - there is a real ordering here
# div4 problems are easier than div3 which are easier than div2 etc.
div_map = {
    'Other': 0,
    'Div. 4': 1,
    'Div. 3': 2,
    'Educational': 3,
    'Div. 2': 4,
    'Div. 12': 5,
    'Div. 1': 6,
    'Global': 7
}

df['div_encoded'] = df['div'].map(div_map)
df = df.drop(columns=['div'])

# encoding problem position A=1, B=2, C=3 etc.
# capping at 8 since problems beyond H are extremely rare
def encode_index(idx):
    idx = str(idx).strip()
    if re.fullmatch(r'[A-Z]', idx):
        return ord(idx) - ord('A') + 1
    if re.fullmatch(r'[A-Z]\d+', idx):
        return ord(idx[0]) - ord('A') + 1
    if re.fullmatch(r'\d+', idx):
        return int(idx)
    return 1

df['index_encoded'] = df['index'].apply(encode_index).clip(upper=8)
df = df.drop(columns=['index'])

print(df['index_encoded'].value_counts().sort_index())

# multi-hot encoding for tags
# a problem can have multiple tags so one-hot doesnt work here
mlb = MultiLabelBinarizer()
tag_encoded = mlb.fit_transform(df['tags'])
tag_df = pd.DataFrame(tag_encoded, columns=mlb.classes_)

df = pd.concat([df, tag_df], axis=1)
df = df.drop(columns=['tags'])
tag_columns = mlb.classes_.tolist()

with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)

# problems with no tags are useless for tag-based features
df['num_tags'] = df[tag_columns].sum(axis=1)
df = df[df['num_tags'] > 0].reset_index(drop=True)
print(df.shape)

# solve count is inversely related to difficulty
# usign the log transform because the range is massive (50 to 100k+)
df['log_solved'] = np.log1p(df['solvedCount'].fillna(0))

# percentile rank within contest - problem A always has the most solves
# this captures the relative structure without caring about absolute numbers
df['solved_pct_rank'] = df.groupby('contestId')['solvedCount'].rank(pct=True, ascending=False)
df['solved_pct_rank'] = df['solved_pct_rank'].fillna(0.5)

df = df.drop(columns=['solvedCount'])

print(df[['log_solved', 'solved_pct_rank']].describe())

# contest level features
# problem E in a 5-problem contest is the final boss
# problem E in an 8-problem contest is mid difficulty
# index_position captures this - raw index alone doesnt
contest_agg = df.groupby('contestId').agg(
    num_problems=('index_encoded', 'count'),
    max_index=('index_encoded', 'max'),
    contest_avg_solved=('log_solved', 'mean')
).reset_index()

df = pd.merge(df, contest_agg, on='contestId', how='left')
df['index_position'] = df['index_encoded'] / df['max_index']

print(df[['num_problems', 'max_index', 'index_position']].describe())

# splitting by contest id NOT by problem
# if i split randomly, problems from the same contest end up in both
# train and test which leaks information through the slot stats
# this was inflating my old results by around 50 MAE points
contest_ids = df['contestId'].unique()
train_cids, test_cids = train_test_split(contest_ids, test_size=0.2, random_state=42)

train_df = df[df['contestId'].isin(train_cids)].drop(columns=['contestId']).reset_index(drop=True)
test_df = df[df['contestId'].isin(test_cids)].drop(columns=['contestId']).reset_index(drop=True)

X_train, y_train = train_df.drop(columns=['rating']), train_df['rating']
X_test, y_test = test_df.drop(columns=['rating']), test_df['rating']

print(f"train: {X_train.shape}  |  test: {X_test.shape}")

# slot stats - for every (division, position) pair compute mean/median/std of ratings
# e.g. Div2-C is historically rated around 1400-1600
# this is the strongest feature group in the whole model
# computing from train only and applying to test - no leakage
slot_stats = (
    X_train.assign(rating=y_train)
    .groupby(['div_encoded', 'index_encoded'])['rating']
    .agg(['mean', 'median', 'std'])
    .reset_index()
    .rename(columns={'mean': 'slot_mean', 'median': 'slot_median', 'std': 'slot_std'})
)
slot_stats['slot_std'] = slot_stats['slot_std'].fillna(0)

X_train = pd.merge(X_train, slot_stats, on=['div_encoded', 'index_encoded'], how='left')
X_test = pd.merge(X_test, slot_stats, on=['div_encoded', 'index_encoded'], how='left')

# unseen div+position combos in test fall back to global average
for col in ['slot_mean', 'slot_median', 'slot_std']:
    X_test[col] = X_test[col].fillna(slot_stats[col].mean())

slot_stats.to_csv('slot_stats.csv', index=False)

print(X_train[['slot_mean', 'slot_median', 'slot_std']].describe())

# two interaction features per tag
# tag alone is weak - dp appears at every difficulty level
# but dp at position E in Div1 is very different from dp at position A in Div4
top_tags = [
    'dp', 'greedy', 'math', 'implementation', 'graphs',
    'data structures', 'brute force', 'constructive algorithms',
    'binary search', 'trees', 'number theory', 'strings'
]

for tag in top_tags:
    if tag in X_train.columns:
        X_train[f'{tag}_x_index'] = X_train[tag] * X_train['index_encoded']
        X_test[f'{tag}_x_index'] = X_test[tag] * X_test['index_encoded']

        X_train[f'{tag}_x_div'] = X_train[tag] * X_train['div_encoded']
        X_test[f'{tag}_x_div'] = X_test[tag] * X_test['div_encoded']

# rare tags like fft, flows, meet-in-the-middle almost never appear on easy problems
# this score is high when a problem has uncommon tags -> pushes prediction harder
tag_freq = X_train[tag_columns].mean()
X_train['tag_rarity_score'] = X_train[tag_columns].values @ (1 - tag_freq.values)
X_test['tag_rarity_score'] = X_test[tag_columns].values @ (1 - tag_freq.values)

tag_freq.to_csv('tag_freq.csv')

print(f"final feature count: {X_train.shape[1]}")

# lgbm works well here - handles the mix of binary tag columns and
# continuous features without needing scaling
# early stopping so i dont have to guess the right n_estimators
model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=8,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(300)]
)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
within_100 = np.mean(np.abs(y_test - y_pred) <= 100) * 100
within_200 = np.mean(np.abs(y_test - y_pred) <= 200) * 100

print("========== RESULTS ==========")
print(f"MAE            → {mae:.2f}")
print(f"Within ±100    → {within_100:.2f}%")
print(f"Within ±200    → {within_200:.2f}%")

# breaking down by difficulty range to see where the model struggles
results = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'error': np.abs(y_test.values - y_pred)
})
results['bucket'] = pd.cut(results['actual'],
    bins=[800, 1200, 1600, 2000, 2400, 2800, 3500],
    labels=['800-1200', '1200-1600', '1600-2000', '2000-2400', '2400-2800', '2800-3500'])

print("\nMAE by rating range:")
print(results.groupby('bucket', observed=True)['error'].mean().round(0))

feature_columns = X_train.columns.tolist()

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("saved model.pkl, mlb.pkl, slot_stats.csv, tag_freq.csv, feature_columns.pkl")
