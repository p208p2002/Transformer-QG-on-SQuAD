from flask import Flask

class ServerMixin():
    def init_server(self):
        self.flask = Flask(__name__)
        
        # add route
        @self.app.route('/')
        def predict():
            # sample_outputs = self.model.generate(
            #     input_ids = input_ids,
            #     attention_mask = attention_mask,
            #     max_length=MAX_INPUT_LENGTH,
            #     early_stopping=True,
            #     temperature=0.85,
            #     do_sample=True,
            #     top_p=0.9,
            #     top_k=10,
            #     num_beams=3,
            #     no_repeat_ngram_size=5,
            #     num_return_sequences=1
            # )
            return 'ok'
        return self.flask
