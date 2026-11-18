#!/usr/bin/env python
# ДЗ 3. Обучение ранжированию — шаблон решения.
#
# Обучение:  ./solution.py --train --model_file=model.bin PATH-TO-DATA
# Сабмишн:   ./solution.py --model_file=model.bin --submission_file=submission.csv PATH-TO-DATA

import argparse
import os
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import math

from catboost import CatBoostRanker, Pool
from sklearn.model_selection import GroupShuffleSplit

SEED = 42


def make_pool(df, features, label, group_id):
    return Pool(
        data=df[features],
        label=df[label],
        group_id=df[group_id]
    )


def dcg(gains, k=10):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))  # позиция i с нуля -> log2(i + 2)


def ndcg(gains, all_gains, k=10):
    ideal = dcg(sorted(all_gains, reverse=True), k)
    return dcg(gains, k) / ideal if ideal > 0 else 0.0


def get_ndcg_score(model, pool, df, k=10):
    preds = model.predict(pool)
    df = df.assign(Prediction=preds)
    grouped = df.groupby('QueryId')
    ndcg_scores = []
    for query_id, group in grouped:
        labels = group['Label'].values
        order = np.argsort(-group['Prediction'].values, kind='stable')
        gains = labels[order]
        ndcg_scores.append(ndcg(gains, labels, k))

    return np.mean(ndcg_scores)


def train(data_dir, model_file):
    data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    features = [column for column in data.columns if column not in ('QueryId', 'Label')]

    # TODO:
    # - отложить часть запросов (целиком, не отдельные строки) и сравнивать варианты по nDCG@10 на них;
    # - обучить модель ранжирования на data[features] и data['Label'], группы — data['QueryId'];
    # - сохранить модель в model_file.

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(data, groups=data['QueryId']))
    train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)

    train_part, val_part = data.iloc[train_idx].copy(), data.iloc[val_idx].copy()

    train_pool = make_pool(train_part, features, 'Label', 'QueryId')
    val_pool = make_pool(val_part, features, 'Label', 'QueryId')

    params = dict(
        loss_function='LambdaMART',
        eval_metric='NDCG:top=10',
        iterations=1000,
        learning_rate=0.05,
        depth=5,
        random_seed=SEED,
        verbose=100
    )

    model = CatBoostRanker(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=200
    )

    model.save_model(model_file)



def predict(data_dir, model_file, submission_file):
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    features = [column for column in test.columns if column not in ('ObjectId', 'QueryId')]

    # TODO: загрузить модель из model_file и посчитать скор для каждой строки test[features].

    model = CatBoostRanker()
    model.load_model(model_file)

    scores = model.predict(test[features])  # замените своими скорами

    submission = pd.DataFrame({'ObjectId': test['ObjectId'], 'Score': scores})
    submission.to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='Learning to rank homework solution')
    parser.add_argument('--train', action='store_true', help='обучить модель и сохранить её в model_file')
    parser.add_argument('--model_file', required=True, help='файл модели')
    parser.add_argument('--submission_file', help='куда записать сабмишн для Kaggle')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()

    # Если в решении есть случайность, зафиксируйте seed: random_seed у CatBoost, seed у LightGBM и т. п.

    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error('без --train нужен --submission_file')
        predict(args.data_dir, args.model_file, args.submission_file)

    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
