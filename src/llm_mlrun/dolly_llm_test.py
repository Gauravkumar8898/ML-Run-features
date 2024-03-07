import mlrun

project = mlrun.get_or_create_project("test-project-2", context="./", user_project=True)

class LanguageModelHandler:
    def __init__(self, model_path):
        self.model_path = model_path
        self.serving_function = mlrun.import_function('function.yaml')
        self.serving_function.add_model(
            'mymodel',
            model_path=self.model_path,
            class_name='HuggingFaceModelServer',
            task="text-generation",
            model_class="GPT2LMHeadModel",
            model_name="lgaalves/gpt2-dolly",
            tokenizer_class="GPT2Tokenizer",
            tokenizer_name="gpt2"
        )

    def run_trainer(self, input_data):
        trainer_run = project.run_function(
            "trainer",
            inputs={"prompt": f"{input_data}"},
        )

    def test_model(self, input_data):
        server = self.serving_function.to_mock_server()
        result = server.test(
            '/v2/models/mymodel',
            body={"inputs": input_data}
        )
        return result['outputs']


# Create or get the MLRun project

# Initialize the LanguageModelHandler class with the model path
lm_handler = LanguageModelHandler('../../../SQL_Python/models/lgaalves_gpt2_dolly')


# Define a function to run the language model handler with input data
# def run_language_model(input_text):
#     output = lm_handler.test_model([input_text])
#     return output


# Example usage of the function with input parameter
# input_text = "I like cake and "
# prediction_output = run_language_model(input_text)
# print(f"Prediction: {prediction_output}")
