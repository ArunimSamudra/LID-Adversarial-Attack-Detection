from textattack import datasets
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.attack_recipes import TextFoolerJin2019, DeepWordBugGao2018, BAEGarg2019
from textattack import Attacker, AttackArgs
import nltk
import ssl
from tqdm import tqdm
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('universal_tagset')

# Load IMDb dataset from CSV
imdb_df = pd.read_csv("IMDB Dataset.csv")

# Convert 'sentiment' column to 1s and 0s
imdb_df['label'] = imdb_df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split data for clean examples
clean_examples = imdb_df.sample(n=333, random_state=42)  # Randomly select 1000 clean examples
adversarial_candidates = clean_examples.copy()  # Use these for generating adversarial examples

# Load pre-trained BERT model for sentiment classification (binary)
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

# Wrap the model for TextAttack
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Set up adversarial attack methods
attacks = {
    # "BAE": BAEGarg2019.build(model_wrapper),
    "DeepWordBug": DeepWordBugGao2018.build(model_wrapper)
    # "TextFooler": TextFoolerJin2019.build(model_wrapper)
}

# Collect clean examples in final data list
data = []
for _, row in clean_examples.iterrows():
    data.append({
        "text": row["review"],
        "label": row["label"],
        "is_clean": True,
        "attack_method": "None",
        "attack_score": None
    })

custom_dataset = [(row['review'], row['label']) for _, row in adversarial_candidates.iterrows()]

# Wrap in a TextAttack Dataset
custom_textattack_dataset = datasets.Dataset(custom_dataset)

# Set up attack and attack arguments
attack = DeepWordBugGao2018.build(model_wrapper)
attack_args = AttackArgs(
    num_examples=333,
    num_workers_per_device=4,
    log_to_csv="custom_attack_log.csv",
    checkpoint_interval=50,
    checkpoint_dir="custom_checkpoints"
)

# Run the attack
attacker = Attacker(attack, custom_textattack_dataset, attack_args)
results = attacker.attack_dataset()

for result in results:
    data.append({
        "text": result.perturbed_result.attacked_text.text,
        "label": result.original_result.ground_truth_output,
        "is_clean": False,
        "attack_method": "DeepWordBug",
        "attack_score": round(result.perturbed_result.score, 4)
    })

# Generate adversarial examples
# for attack_name, attack in attacks.items():
#     successful_adv = 0
#     # for _, row in tqdm(adversarial_candidates.iterrows(), desc=f"Generating {attack_name} adversarial examples"):
#     #     if successful_adv >= 333:  # Approximately 1000 / 3 for each attack
#     #         break
#     #     try:
#     #         adversarial_example = attack.attack(row["review"], row["label"])
#     #         data.append({
#     #             "text": adversarial_example.perturbed_result.attacked_text.text,
#     #             "label": row["label"],
#     #             "is_clean": False,
#     #             "attack_method": attack_name,
#     #             "attack_score":  round(adversarial_example.perturbed_result.score, 4)
#     #         })
#     #         successful_adv += 1
#     #         print(f"Generated new adversarial example with {attack_name}")
#     #     except Exception as e:
#     #         print(f"Failed to generate adversarial example with {attack_name}: {e}")
#     for _, row in tqdm(adversarial_candidates.iterrows(), desc=f"Generating {attack_name} adversarial examples"):
#         try:
#             adversarial_example = attack.attack(row["review"], row["label"])
#             data.append({
#                 "text": adversarial_example.perturbed_result.attacked_text.text,
#                 "label": row["label"],
#                 "is_clean": False,
#                 "attack_method": attack_name,
#                 "attack_score":  round(adversarial_example.perturbed_result.score, 4)
#             })
#             print(f"Generated new adversarial example with {attack_name}")
#         except Exception as e:
#             print(f"Failed to generate adversarial example with {attack_name}: {e}")

# Save final dataset
df = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)
df.to_csv("imdb_with_adversarial_examples.csv", index=False)
print("Dataset with adversarial examples created and saved.")