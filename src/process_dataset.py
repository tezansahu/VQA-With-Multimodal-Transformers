import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os
import yaml
from typing import Text

def processDaquarDataset(config_path: Text):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    image_pattern = re.compile("( (in |on |of )?(the |this )?(image\d*) \?)")

    with open(os.path.join(config["data"]["dataset_folder"], config["data"]["all_qa_pairs_file"])) as f:
        qa_data = [x.replace("\n", "") for x in f.readlines()]

    # with open("train_images_list.txt") as f:
    #     train_imgs = [x.replace("\n", "") for x in f.readlines()]

    # with open("test_images_list.txt") as f:
    #     test_imgs = [x.replace("\n", "") for x in f.readlines()]

    df = pd.DataFrame({config["data"]["question_col"]: [], config["data"]["answer_col"]: [], config["data"]["image_col"]:[]})

    for i in range(0, len(qa_data), 2):
        img_id = image_pattern.findall(qa_data[i])[0][3]
        question = qa_data[i].replace(image_pattern.findall(qa_data[i])[0][0], "")
        record = {
            config["data"]["question_col"]: question,
            config["data"]["answer_col"]: qa_data[i+1],
            config["data"]["image_col"]: img_id,
        }
        df = df.append(record, ignore_index=True)

    answer_space = []
    for ans in df.answer.to_list():
        answer_space = answer_space + [ans] if "," not in ans else answer_space + ans.replace(" ", "").split(",") 

    answer_space = list(set(answer_space))
    answer_space.sort()
    with open(os.path.join(config["data"]["dataset_folder"], config["data"]["answer_space"]), "w") as f:
        f.writelines("\n".join(answer_space))

    # train_df = df[df.image_id.isin(train_imgs)]
    # test_df = df[df.image_id.isin(test_imgs)]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(os.path.join(config["data"]["dataset_folder"], config["data"]["train_dataset"]), index=None)
    test_df.to_csv(os.path.join(config["data"]["dataset_folder"], config["data"]["eval_dataset"]), index=None)

    df.to_csv("data.csv", index=None)