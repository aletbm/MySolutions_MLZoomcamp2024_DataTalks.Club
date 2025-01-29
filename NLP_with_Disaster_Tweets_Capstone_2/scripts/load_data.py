import kagglehub
import shutil
import os

def download_dataset():
    for dir in ["kaggle", "kaggle/input", "kaggle/input/nlp-getting-started", "kaggle/input/479k-english-words", "kaggle/input/english-word-frequency", "kaggle/working", "kaggle/working/models", "kaggle/working/histories"]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    nlp_getting_started_path = kagglehub.competition_download('nlp-getting-started', force_download=True)
    rtatman_english_word_frequency_path = kagglehub.dataset_download('rtatman/english-word-frequency', force_download=True)
    yk1598_479k_english_words_path = kagglehub.dataset_download('yk1598/479k-english-words', force_download=True)

    target_dir = "kaggle/input/nlp-getting-started/"
    file_names = os.listdir(nlp_getting_started_path)
    for file_name in file_names:
        shutil.move(os.path.join(nlp_getting_started_path, file_name), target_dir)

    target_dir = "kaggle/input/479k-english-words/"
    file_names = os.listdir(yk1598_479k_english_words_path)
    for file_name in file_names:
        shutil.move(os.path.join(yk1598_479k_english_words_path, file_name), target_dir)

    target_dir = "kaggle/input/english-word-frequency/"
    file_names = os.listdir(rtatman_english_word_frequency_path)
    for file_name in file_names:
        shutil.move(os.path.join(rtatman_english_word_frequency_path, file_name), target_dir)
        
    print("Data source import complete.")
    return