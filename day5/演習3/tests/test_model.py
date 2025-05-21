import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"

def test_compare_with_baseline_model(train_model):
    """現在のモデルがベースラインモデルより性能が悪化していないことを検証"""
    current_model, X_test, y_test = train_model
    
    # ベースラインモデルのパス（過去のモデルが保存されている場所）
    baseline_model_path = os.path.join(os.path.dirname(__file__), "../models/baseline_titanic_model.pkl")
    
    # ベースラインモデルが存在しない場合は現在のモデルをベースラインとして保存
    if not os.path.exists(baseline_model_path):
        with open(baseline_model_path, "wb") as f:
            pickle.dump(current_model, f)
        pytest.skip("ベースラインモデルが存在しないため、現在のモデルをベースラインとして保存しました")
    
    # ベースラインモデルをロード
    with open(baseline_model_path, "rb") as f:
        baseline_model = pickle.load(f)
    
    # 両方のモデルで予測して精度を計算
    current_predictions = current_model.predict(X_test)
    baseline_predictions = baseline_model.predict(X_test)
    
    current_accuracy = accuracy_score(y_test, current_predictions)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    
    # 現在のモデルがベースラインモデル以上の精度であることを確認
    # 許容誤差5%以内なら合格
    assert current_accuracy >= baseline_accuracy * 0.95, \
        f"現在のモデル精度({current_accuracy:.4f})がベースラインモデル精度({baseline_accuracy:.4f})の95%未満です"

def test_condo_matrix(train_model):
    """混同行列を検証"""
    model, X_test, y_test = train_model

    # 予測
    y_pred = model.predict(X_test)

    # 混同行列の計算
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # メトリクス計算
    precision = tp / (tp+fp) if (tp + fp) > 0 else 0
    recall = tp / (tp+fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"混同行列:\n{cm}")
    print(f"真陽性(TP): {tp}, 真陰性(TN): {tn}")
    print(f"偽陽性(FP): {fp}, 偽陰性(FN): {fn}")
    print(f"精度(Precision): {precision:.4f}")
    print(f"再現率(Recall): {recall:.4f}")
    print(f"F1スコア: {f1:.4f}")
    
    # 精度、再現率、F1スコアの閾値を設定
    precision_threshold = 0.6
    recall_threshold = 0.6
    f1_threshold = 0.6
    
    # 精度、再現率、F1スコアが閾値を満たすことを確認
    assert precision >= precision_threshold, f"精度が閾値を下回っています: {precision:.4f}"
    assert recall >= recall_threshold, f"再現率が閾値を下回っています: {recall:.4f}"
    assert f1 >= f1_threshold, f"F1スコアが閾値を下回っています: {f1:.4f}"
