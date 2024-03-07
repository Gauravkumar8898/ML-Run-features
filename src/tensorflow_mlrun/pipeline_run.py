import mlrun


def pipeline():
    project = mlrun.get_or_create_project("tensorflow-mlrun", context="./", user_project=True)

    data_gen_fn = project.set_function(
        "data.py",
        name="data",
        kind="job",
        image="mlrun/mlrun",
        handler="wine_data_generator",
    )

    project.save()  # save the project with the latest config

    gen_data_run = project.run_function("data", local=True)

    trainer = project.set_function(
        "trainer.py", name="trainer", kind="job", image="mlrun/mlrun", handler="train"
    )
    trainer_run = project.run_function(
        "trainer",
        inputs={
            "dataset": gen_data_run.outputs["dataset"]
        },
        local=True,
    )

    serving_fn = mlrun.new_function(
        "serving",
        image="mlrun/mlrun",
        kind="serving",
        requirements=["scikit-learn~=1.3.0", "tensorflow==2.15.0"],
    )

    serving_fn.add_model(
        "wine-classifier",
        model_path=trainer_run.outputs["model"],
        class_name="mlrun.frameworks.tf_keras.TFKerasModelServer",
    )

    # Plot the serving graph topology
    serving_fn.spec.graph.plot(rankdir="LR")
    server = serving_fn.to_mock_server()
