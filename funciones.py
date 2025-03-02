import argparse
import time
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score


def argumentos():
    parser = argparse.ArgumentParser(description='Wine Classifier Training with Gradient Boosting.')
    parser.add_argument('--nombre_job', type=str, help='Job name for MLflow experiment.', required=False, default="wine-classification")
    parser.add_argument('--n_estimators_list', nargs='+', type=int, help='List of n_estimators values to try.', required=True)
    parser.add_argument('--learning_rate_list', nargs='+', type=float, help='List of learning_rate values to try.', required=True)
    parser.add_argument('--max_depth_list', nargs='+', type=int, help='List of max_depth values to try.', required=True)
    parser.add_argument('--subsample_list', nargs='+', type=float, help='List of subsample values to try.', required=True)
    return parser.parse_args()


def load_dataset():
    wine = load_wine()
    df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
    df['target'] = wine['target']
    return df


def data_treatment(df, test_size=0.2):
    # Split data 
    x_raw = df.drop('target', axis=1)
    y_raw = df['target']
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=test_size, random_state=42, stratify=y_raw)
    return x_train, x_test, y_train, y_test


def mlflow_tracking(nombre_job, x_train, x_test, y_train, y_test, n_estimators_list, learning_rate_list, max_depth_list, subsample_list):
    mlflow.set_experiment(nombre_job)
    for n_estimators in n_estimators_list:
        for learning_rate in learning_rate_list:
            for max_depth in max_depth_list:
                for subsample in subsample_list:
                    with mlflow.start_run() as run:
                        clf = GradientBoostingClassifier(n_estimators=n_estimators,
                                                         learning_rate=learning_rate,
                                                         max_depth=max_depth,
                                                         subsample=subsample,
                                                         random_state=42)

                        preprocessor = Pipeline(steps=[('scaler', StandardScaler())])

                        model = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('classifier', clf)])

                        
                        model.fit(x_train, y_train)
                        accuracy_train = accuracy_score(y_train, model.predict(x_train))
                        accuracy_test = accuracy_score(y_test, model.predict(x_test))

                        
                        mlflow.log_metric('accuracy_train', accuracy_train)
                        mlflow.log_metric('accuracy_test', accuracy_test)
                        mlflow.log_param('n_estimators', n_estimators)
                        mlflow.log_param('learning_rate', learning_rate)
                        mlflow.log_param('max_depth', max_depth)
                        mlflow.log_param('subsample', subsample)
                        mlflow.log_param('test_size', 0.2)  

                        
                        mlflow.sklearn.log_model(model, 'wine_gb_model')
    print("All models have been trained and logged.")


def main():
    args = argumentos()


    df = load_dataset()
    x_train, x_test, y_train, y_test = data_treatment(df)

    
    mlflow_tracking(args.nombre_job, x_train, x_test, y_train, y_test, args.n_estimators_list, args.learning_rate_list, args.max_depth_list, args.subsample_list)

if __name__ == '__main__':
    main()
