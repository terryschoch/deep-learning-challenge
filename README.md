# Module 21: deep-learning-challenge

## Alphabet Soup Funding Analysis Report

## Overview 
### The Objective
The nonprofit foundation Alphabet Soup has asked us for a tool that can help it select the applicants for funding with the best chance of success in their ventures. We will be applying our knowledge of machine learning and neural networks and using the features in a dataset provided to us by Alphabet Soup to create a binary classifier that can predict whether applicants will be successful if funded.

### The Data
As mentioned above, Alphabet Soup’s business team has shared a CSV containing more than 34,000 organizations that have received funding from the foundation over the years. Within this dataset are a number of columns that capture metadata about each organization; they include:
    - **The Features**:
        - **APPLICATION_TYPE**: Alphabet Soup application type
        - **AFFILIATION**: Affiliated sector of industry
        - **CLASSIFICATION**: Government organization classification
        - **USE_CASE**: Use case for funding
        - **ORGANIZATION**: Organization type
        - **STATUS**: Active status
        - **INCOME_AMT**: Income classification
        - **SPECIAL_CONSIDERATIONS**: Special considerations for application
        - **ASK_AMT**: Funding amount requested        
    - **The Target**
        - **IS_SUCCESSFUL**: Was the money used effectively
    - **The Extras**:
        - **EIN** and **NAME**: Identification columns


### Step 1: The Preprocessing
After loading and conducting an initial review of the ***charity_data.csv***, identifying the *features* and *target* became the first step to complete. As noted above, in this case the ***IS_SUCCESSFUL*** column was deemed to be our target. The ***EIN*** and ***NAME*** columns were considered to be extraneous information and columns that we could remove from the dataset by way of the .drop function. The remaining columns were then designated as our features. 

    - *What variable(s) are the target(s) for your model?*
        - **IS_SUCCESSFUL**
        
    - *What variable(s) are the features for your model?*
        - **APPLICATION_TYPE**
        - **AFFILIATION**
        - **CLASSIFICATION**
        - **USE_CASE**
        - **ORGANIZATION**
        - **STATUS**
        - **INCOME_AMT**
        - **SPECIAL_CONSIDERATIONS**
        - **ASK_AMT**

    - *What variable(s) should be removed from the input data because they are neither targets nor features?*
        - **EIN** and **NAME**

Once identified and the extraneous columns dropped, the remaining columns were further cleaned by determining the number of unique values for each column through the use of the *.nunique()* function and binning outlier values for certain columns into catchall bins labelled "Other". In this case, this was achieved by way of applying *.value_counts* on the ***APPLICATION_TYPE*** and ***CLASSIFICATION*** columns and creating respective cut off values for each (<500 and <300) where the value would be replaced with the "Other" designation with the *.replace* function.

Upon confirmating the Dtypes for each column with *.info()*, we next created a categorical variable list of columns which were converted and encoded using the *pd.get_dummies* function. The resultant ***encoded_df*** was then merged with the original ***application_df***, dropping the original column values in the process.

With our data now cleaned, it was seperated into ***X*** and ***y*** based on our features and target designations. It was next split into training (***X_train, y_train***) and testing (***X_test, y_test***) datasets using the *train_test_split* function. *StandardScaler()* instances were instantiated and the ***X_train*** data was fit to the scaler and then scaled concurrently but seperate from the ***X_test*** data using the *.transform* function.


### Step 2: The Deep Learning

A model was now defined ***nn*** as well as its number of input features (44) and the number of hidden nodes (8 and 5) for each layer as well as their respective activiation functions (*relu, sigmoid*). A model *.summary()* was returned indicating 411 trainable parameters would be applied. The model was then compiled and a callback log initiated to record the model's weights at certain intervals. Finally, the model was fit and trained using the dataset for a duration of 30 epochs. The *.evaluate* function revealed a loss of 55.7% (0.5573) and accuracy score of 72.7% (0.7266).



We next trained and evaluated a second *logistic regression model* with resampled data which we acquired by using *RandomOverSampler* from the *imbalanced-learn library*. The model was fit to the resampled data and predictions again made followed by a performance evaluation via calculating the accuracy score, generating a confusion matrix and printing the classification report. The same question for the first model was posed and answered for the resampled model.


### Step 3: The Optimization


## The Results
### Deep Learning Model Scores:
**Model 1 (nn):**
  - Input Features: **44**
  - Hidden Layers: **2**
    - L1: **8** nodes, *relu* act
    - L2: **5** nodes, *relu* act
  - Accuracy Score: **72.7%**
  - Loss Score: **55.7%**
     Accuracy Image / Loss Image

**Model 2 (opta):**
  - Input Features: **44**
  - Hidden Layers: **2**
    - L1: **8** nodes, *relu* act
    - L2: **5** nodes, *relu* act
  - Accuracy Score: **72.7%**
  - Loss Score: **55.7%**
     Accuracy Image / Loss Image

**Model 3 (optb):**
  - Input Features: **44**
  - Hidden Layers: **2**
    - L1: **8** nodes, *relu* act
    - L2: **5** nodes, *relu* act
  - Accuracy Score: **72.7%**
  - Loss Score: **55.7%**
     Accuracy Image / Loss Image

**Model 4 (optc):**
  - Input Features: **44**
  - Hidden Layers: **2**
    - L1: **8** nodes, *relu* act
    - L2: **5** nodes, *relu* act
  - Accuracy Score: **72.7%**
  - Loss Score: **55.7%**
     Accuracy Image / Loss Image
    

## Summary
Based on the original test data, the logistic regression model did a great job by achieving a 99.2% balanced accuracy score. In regard to predicting healthy loans, it reached 100% across the board in precision (TP/(TP+FP)), recall (TP/(TP+FN)) and F1 (2(PrecisionRecall)/(Precision+Recall)). However, the precision accuracy dropped to 87% on High Risk Loans which was a significant decline relatively speaking. The recall rate also dropped to 89% and the F1 score for high risk loans predictions was just 88% as a result.

This represented a fairly signficant miss rate in the prediction model in regard to accurately predicting high risk loans, particularily the 89% recall rate was concerning. But given the cost opportunity of profit gained by offering succesful loans vs the cost of covering a loss for a failed loan, the model still did a reasonable job overall and could be considered viable given the macro average for the f1-score was 94% and the weighted average an even better 99%.

Conversely, the resampled data model achieved a slighty better 99.5% balanced accuracy rate and an identical 100% across the board in precision, recall and F1 for Healthy Loans. However, while the precision score again dropped to 87%, the recall score was a perfect 100% on this model and the F1 score was 93% as a result. While the miss rate was still significant and a fair number of false positives were registered for High Risk loans as shown by the precission score, that there were no unidentified high risk loans was a significant improvement. The opportunity cost in this scenario was such that it would likely greatly mitigate the signficance of the precision inaccuracy. 

Moreover, the far superior recall rate with the resampled data model indicated that overall clearly this should be the model recommendeded for use.