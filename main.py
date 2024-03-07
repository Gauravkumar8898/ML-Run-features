from src.llm_mlrun.dolly_llm_test import LanguageModelHandler

if __name__ == '__main__':
    handler = LanguageModelHandler(model_path='./models/lgaalves_gpt2_dolly')
    input_text = input("Enter a incomplete sentence : ")
    output = handler.test_model([input_text])
    print(f"Prediction: {output}")
