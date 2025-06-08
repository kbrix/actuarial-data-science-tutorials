# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    model_name = r"distilbert-base-multilingual-cased"
    save_path = r"C:\Users\brix\Downloads\distilbert-base-multilingual-cased-onnx"

    # Export the model with last_hidden_state
    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save ONNX model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
