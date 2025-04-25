import os
import yaml
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

# -----------------------------------------------------------------------------

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# -----------------------------------------------------------------------------

def import_params(path: str) -> dict:
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('SUCCESSFULLY parameters import from params.yaml')
        return params

    except Exception as e:
        logger.error('something wrong with params.yaml', exe_info=True)

# -----------------------------------------------------------------------------
def data_loading(path1: str, path2: str) -> pd.DataFrame:
    try:
        train_data = pd.read_csv(path1)
        test_data = pd.read_csv(path2)

        logger.debug("SUCCESSFULLY data loaded")
        return train_data, test_data
    
    except Exception as e:
        logger.error('Problem occur during data loading', exc_info=True)

# -----------------------------------------------------------------------------

def data_spliting(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    try:
        x_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        x_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        logger.debug('SUCCESSFULLY data split')

        return x_train, x_test, y_train, y_test
    
    except Exception as e:
        logger.error('Error occur during data spliting')

# -----------------------------------------------------------------------------

def tfidf_vertor(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, max_feature: int) -> pd.DataFrame:
    try:
        vectorizer = TfidfTransformer(max_features=max_feature)

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('SUCCESSFULLY Bag of Words applied and data transformed')
        return train_df, test_df
    
    except Exception as e:
        logger.error('Error during Bag of Words transformation:', exc_info=True)
        
# -----------------------------------------------------------------------------

def storing(x_train_bow: pd.DataFrame, x_test_bow: pd.DataFrame, data_path: str) -> None:
    try:
        x_train_bow.to_csv(os.path.join(data_path, 'train_transformed.csv'), index=False)
        x_test_bow.to_csv(os.path.join(data_path, 'test_transformed.csv'), index=False)

        logger.debug('SUCCESSFULLY data stored')

    except Exception as e:
        logger.error('Error occur during the storing process')

# -----------------------------------------------------------------------------

def main():
    try:
        params = import_params('params.yaml')
        train_data, test_data = data_loading('./data/raw/train.csv', './data/raw/test.csv')

        x_train, x_test, y_train, y_test = data_spliting(train_data, test_data)
        
        max_feature = params['feature_engineering']['max_features']
        x_train_bow, x_test_bow = tfidf_vertor(x_train, x_test, y_train, y_test, max_feature)

        data_path = os.path.join("./data", "processed",)
        os.makedirs(data_path, exist_ok=True)
        storing(x_train_bow, x_test_bow, data_path)

        logger.debug('main() run SUCCESSFULLY')

    except Exception as e:
        logger.error('Error occur in main() funciton', exc_info=True)

# -----------------------------------------------------------------------------
    
if __name__ == '__main__':
    main()

# -----------------------------------------------------------------------------