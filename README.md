

# Parliamanet Election NLP

## About
NLP Analyse of election machine texts from the Finnish parliament elections 2015.

The data used for the prediction comes from avoindata.fi: https://www.avoindata.fi/data/fi/dataset/eduskuntavaalien-2015-ylen-vaalikoneen-vastaukset-ja-ehdokkaiden-taustatiedot

The result of the analysis is that the model could make party predictions of any given candidates "I want to work towards"- statemanet with approx. 44% accuracy on the test data. A complete report and statistics can be found under reports: https://github.com/chpatola/election_nlp/tree/master/reports

## How to use

1. Clone this repository to your own computer
2. cd to the repository
3. Create a new conda environment where <env> is the name you want to give the new environment. Then activate the enivronment
```
conda create --name <env> --file requirement.txt
  
conda activate <env>  
```
 
4. Run the code from the file predict_model.py.
  
  ```
  python predict_model.py
  ```
5. The programme will print some output of the analysis but you will also find results in the reports folder: https://github.com/chpatola/election_nlp/tree/master/reports