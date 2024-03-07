import mlrun
def dynamic_run():
    project = mlrun.get_or_create_project("tensorflow-mlrun", "./", user_project=True, init_git=False)
    function = project.set_function(func="/home/knoldus/Desktop/MLrun/src/pipeline/transformers_pipeline.py",name="newest-runner", kind="job",image="mlrun/mlrun")
    gen_data_run = project.run_function("new-runner",inputs={"context":project.context} ,local=True)
    project.save()

