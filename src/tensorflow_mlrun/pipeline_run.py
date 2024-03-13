import mlrun


def pipeline():
    project = mlrun.get_or_create_project("tensorflow-mlrun", context="./", user_project=True)

    data_gen_fn = project.set_function(
        "data.py",
        name="data",
        kind="job",
        image="mlrun/mlrun",
        handler="diabetes_data_generator",
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


