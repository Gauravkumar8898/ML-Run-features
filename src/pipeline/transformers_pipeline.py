import mlrun
from transformers import pipeline
import logging

class Mlrun_Pipeline_Transformer:
    def __init__(self):
        self.text_pipeline= pipeline("sentiment-analysis")
        self.project = mlrun.get_or_create_project("tensorflow-mlrun", "./", user_project=True, init_git=True)
        # login()
        logging.basicConfig(level=logging.INFO)


    @mlrun.handler()
    # @staticmethod
    def runner(self,context):

        sentiment_analyzer = pipeline("sentiment-analysis")
        sentences_to_analyze = ["I sure would like to see a resurrection of a up dated Seahunt series with the tech they have today it would bring back the kid excitement in me.I grew up on black and white TV and Seahunt with Gunsmoke were my hero's every week.You have my vote for a comeback of a new sea hunt.We need a change of pace in TV and this would work for a world of under water adventure.Oh by the way thank you for an outlet like this to view many viewpoints about TV and the many movies.So any ole way I believe I've got what I wanna say.Would be nice to read some more plus points about sea hunt.If my rhymes would be 10 lines would you let me submit,or leave me out to be in doubt and have me to quit,If this is so then I must go so lets do it.",
                                "Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his parents are fighting all the time.his movie is slower than a soap opera... and suddenly, Jake decides to become Rambo and kill the zombie.OK, first of all when you're going to make a film you must Decide if its a thriller or a drama! As a drama the movie is watchable. Parents are divorcing & arguing like in real life. And then we have Jake with his closet which totally ruins all the film! I expected to see a BOOGEYMAN similar movie, and instead i watched a drama with some meaningless thriller spots.3 out of 10 just for the well playing parents & descent dialogs. As for the shots with Jake: just ignore them.",
                                "AWWWW, I just love this movie to bits. Me and my cousins enjoy this movie a lot and I am just such a HUGE FAN!!! I hope they bring the TV series out on DVD soon. Come to mention it, I have not see the TV show in a LONG time. Such geart times! Where I come from Australia The Chipmunk Adventure is only known by people in their late teens and adult years which is kinda sad because the young kids don't know what there missing.The songs in this film are ace the ones I love the most Boys/girls of rock n'roll, Diamond Dolls and the song that ls sure to make you want to cry My Mother.This film is sure to excite both young and old GET THE CHIPMUNK ADVENTURE TODAY!!! 10 out of 10, such an excellent movie."]
        for text in sentences_to_analyze:
            result = sentiment_analyzer(text)
            print(f"Text: {text}, Sentiment: {result[0]['label']} with confidence: {result[0]['score']}")
            context.log_result(key=f"Text: {text}",
                               value=f"Sentiment: {result[0]['label']} with confidence: {result[0]['score']}")


    def static_runner(self):
        context=mlrun.get_or_create_ctx('transformers-mlflow')
        sentiment_analyzer = pipeline("sentiment-analysis")
        sentences_to_analyze = ["I sure would like to see a resurrection of a up dated Seahunt series with the tech they have today it would bring back the kid excitement in me.I grew up on black and white TV and Seahunt with Gunsmoke were my hero's every week.You have my vote for a comeback of a new sea hunt.We need a change of pace in TV and this would work for a world of under water adventure.Oh by the way thank you for an outlet like this to view many viewpoints about TV and the many movies.So any ole way I believe I've got what I wanna say.Would be nice to read some more plus points about sea hunt.If my rhymes would be 10 lines would you let me submit,or leave me out to be in doubt and have me to quit,If this is so then I must go so lets do it.",
                                "Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his parents are fighting all the time.his movie is slower than a soap opera... and suddenly, Jake decides to become Rambo and kill the zombie.OK, first of all when you're going to make a film you must Decide if its a thriller or a drama! As a drama the movie is watchable. Parents are divorcing & arguing like in real life. And then we have Jake with his closet which totally ruins all the film! I expected to see a BOOGEYMAN similar movie, and instead i watched a drama with some meaningless thriller spots.3 out of 10 just for the well playing parents & descent dialogs. As for the shots with Jake: just ignore them.",
                                "AWWWW, I just love this movie to bits. Me and my cousins enjoy this movie a lot and I am just such a HUGE FAN!!! I hope they bring the TV series out on DVD soon. Come to mention it, I have not see the TV show in a LONG time. Such geart times! Where I come from Australia The Chipmunk Adventure is only known by people in their late teens and adult years which is kinda sad because the young kids don't know what there missing.The songs in this film are ace the ones I love the most Boys/girls of rock n'roll, Diamond Dolls and the song that ls sure to make you want to cry My Mother.This film is sure to excite both young and old GET THE CHIPMUNK ADVENTURE TODAY!!! 10 out of 10, such an excellent movie."]
        for text in sentences_to_analyze:
            result = sentiment_analyzer(text)
            print(f"Text: {text}, Sentiment: {result[0]['label']} with confidence: {result[0]['score']}")
            context.log_result(key=f"Text: {text}",
                               value=f"Sentiment: {result[0]['label']} with confidence: {result[0]['score']}")
    # @app.route('/predict', methods=['POST'])
    # def predict(self):
    #
    #     if request.method == 'POST':
    #         data = request.get_json()
    #         text = data.get('text', '')
    #         for sentence in text:
    #             result=self.text_pipeline(sentence)
    #             print(f"Text: {text}, Sentiment: {result[0]['label']} with confidence: {result[0]['score']}")
    #             self.context.log_result(key=f"Text: {text}",
    #                                value=f"Sentiment: {result[0]['label']} with confidence: {result[0]['score']}")
    #
    #
    #         return jsonify(result)











# obj=Mlrun_Pipeline_Transformer()
# obj.runner()
