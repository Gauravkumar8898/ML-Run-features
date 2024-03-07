import mlrun


project = mlrun.get_or_create_project("openai", context="./", user_project=True)
# Configure the MLRun function
openai_predict_func = project.set_function(
    func="open_ai.py",
    name="test2",
    kind="job",
    image="mlrun/mlrun",
    handler="openais",
    requirements=["openai"]
)
# comp_func = mlrun.import_function(openais(project.context, "Hello"))
trainer_run = project.run_function(
    "test2",
    inputs={
        "prompt": "how are you?",
        "context": project.context
    },
    # params={"n_estimators": 100, "learning_rate": 1e-1, "max_depth": 3},
    local=True,
)
project.save()
# project.run_function("trainer")
