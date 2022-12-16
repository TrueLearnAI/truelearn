import pandas as pd

input_path = "C:\\Users\\in4maniac\\Documents\\Data\\x5gon\\Truelearn2\\training_data\\final_data_top_10_topics_20_sessions_75_engagement_all_topics_pre_swx_pr_0.8_cos_0.2\\results\\semantic_truelearn_neg\\_test\\model_results.json"

with open(input_path) as infile:
    records = pd.read_json(input_path)

records = records[['session', 'accuracy', 'precision', 'recall', 'f1', 'num_events',
                   'num_topics_rate', 'positive_rate', 'predict_positive_rate',
                   'change_label_rate', 'num_topics', 'num_user_topics']]

records.to_csv("C:\\Users\\in4maniac\\Documents\\Data\\x5gon\\Truelearn2\\training_data\\final_data_top_10_topics_20_sessions_75_engagement_all_topics_pre_swx_pr_0.8_cos_0.2\\results\\semantic_truelearn_neg\\_test\\model_results.csv", index=False)
