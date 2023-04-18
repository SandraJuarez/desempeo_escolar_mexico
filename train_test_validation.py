from sklearn.model_selection import train_test_split

def split_data(X,y):
    X_train, X_test, y_train, y_test=train_test_split(
        X,y, test_size=1/3, random_state=1, shuffle=True, stratify=None)
    X_test, X_v, y_test, y_v=train_test_split(
        X_test,y_test, test_size=1/3, random_state=1, shuffle=True, stratify=None)
    return X_train, X_test, X_v,y_train, y_test, y_v

if __name__=="__MAIN__":
    split_data()