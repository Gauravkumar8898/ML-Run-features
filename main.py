import logging
import mlrun
from src.pipeline.transformers_pipeline import Mlrun_Pipeline_Transformer
from src.pipeline.pipeline_run import dynamic_run


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    choice=int(input(('Enter a number:')))
    if choice==1:
        logging.info('Logging static predictions to the mlrun ui....')
        obj=Mlrun_Pipeline_Transformer()
        obj.static_runner()

    if choice==2:
        logging.info('Sending dynamic data via postman and serving flask application to mlrun......')
        dynamic_run()


