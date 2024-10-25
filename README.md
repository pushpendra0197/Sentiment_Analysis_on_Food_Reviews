I have created  Restaurant Food Review Model based on output Liked or Disliked after clean the data by  using different NLP technqiues stopwords removal,punctuation removal,special character removal. And trained model with all classification supervised machine learning algorithm and chose best Accuracy given algorithm and deployed over Modelbit deployment platform.
platform so model can accessible from anywhere. Kindly install modelbit in your IDE(jupyter,colab) by using (pip install modelbit) import modelbit module.

Command for Curl-

curl -s -XPOST "https://pushpendradhamanya.us-east-2.aws.modelbit.com/v1/Nlp_Food_Review_Predictor/latest" -d '{"data": Review}' | json_pp



Python-

modelbit.get_inference(
  region="us-east-2.aws",
  workspace="pushpendradhamanya",
  deployment="Nlp_Food_Review_Predictor",
  data=Review
)

Enter your review in string format in data field. Then Model will give you output as Liked or Disliked.
