from calc_distance import calc_distance
from experiments import *
import probing

if __name__ == '__main__':

    categories = ["Books", "Electronics", "Movies_and_TV", "CDs_and_Vinyl", "Clothing_Shoes_and_Jewelry",
                  "Home_and_Kitchen", "Kindle_Store", "Sports_and_Outdoors", "Cell_Phones_and_Accessories",
                  "Health_and_Personal_Care", "Toys_and_Games", "Video_Games", "Tools_and_Home_Improvement",
                  "Beauty", "Apps_for_Android", "Office_Products", "Pet_Supplies", "Automotive",
                  "Grocery_and_Gourmet_Food", "Patio_Lawn_and_Garden", "Baby", "Digital_Music", "Amazon_Instant_Video"]





    clf_for_category = fill_category_classifier_dict(categories, 1, 10, "LinearSVC")
    in_domain_results = fill_in_domain_performance_dict(categories, clf_for_category, 1, 10)
    cross_domain_results = fill_cross_domain_performance_dict(categories, clf_for_category, 1, 10)
    relative_loss = calc_relative_loss(categories, in_domain_results, cross_domain_results)
    cls_a_distance = calc_distance(categories, 1, 10, "cls", "LinearSVC")
    probing.calc_src_stdev("Beauty", categories, relative_loss)
    top_domain_features = probing.get_top_domain_features("Beauty", "Automotive", 100, 1, 10)
    top_sentiment_features = probing.get_top_sentiment_features("Beauty", 100, 1, 10)
    probing.check_feature_overlap(top_domain_features, top_sentiment_features)




