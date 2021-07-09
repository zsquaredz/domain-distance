# domain-distance
Repository for DA experiments

How to construct a dataset from McAuley's files:

for c in categories:
    rdp = ReviewDataProvider(c)
    reviews, labels = rdp.construct_dataset(1000)
