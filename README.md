# domain-distance
Repository for DA experiments


## How to construct a dataset from McAuley's files
Locate the files under "data" directory.

For example, the "reviews_Apps_for_Android_5.json.gz" is corresponding to McAuley's "Apps_for_Android" review dump.

Running the following in main.py:
```
from create_dataset import ReviewDataProvider
if __name__ == '__main__':
    category = "Apps_for_Android"
    reviews_per_class = 1000
    rdp = ReviewDataProvider(category)
    reviews, labels = rdp.construct_dataset(reviews_per_class)
```
Will be resulted in a 2000 reviews under "data/X_Apps_for_Android_5.pkl" and 2000 labels under "y_Apps_for_Android_5.pkl"

## How to load an existing dataset
Running the following in main.py:
```
import utils
if __name__ == '__main__':
    category = "Apps_for_Android"
    X_train, y_train, X_test, y_test = utils.load_existing_dataset(category)
```
Will be resulted in balanced train and test sets, equel in size.
