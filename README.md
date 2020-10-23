# credit_score

This is an assessment for the Food Panda technical test with the following questions

Please follow the steps described below:
1) Sign up at kaggle.com (if you are not yet a user) and go to https://www.kaggle.com/c/GiveMeSomeCredit, read through the description of the challenge, download the data files
2) Objective: solve the above challenge.
3) Please submit your code (preferably in Python and with a github repo or a link to your source code, with .git folder) and also answer the following questions:
---
1. Tell us how you validate your model, which, and why you chose such evaluation technique(s).
  - Experiment with a few different families of models:
    - Why? To make sure I choose the best class of algorithms to solve this
  - Conduct a 5-fold split:
    - Why? To ensure it is not a one-off data split stroke of luck in good model performance by averaging performance
  - Find the model with the best AUC:
    - Why? Use the AUC metric to validate the best class of algorithm
  - Hyper-parameter tuning:
    - Why? Optimise this model
---
2. What is AUC? 
- AUC is the area under the ROC curve, with axes TPR=Recall, FPR=1-Precision, at different thresholds. Therefore the higher the AUC, the better the model is generally. 
- However, if you have a lot of negative examples as in this dataset, it does not get punished. This will be illustrated in the summary of Question 5.
---

3. Why do you think AUC was used as the evaluation metric for such a problem? 
- As a competitive metric, it removes the need for thresholds in deciding who has the best model - which is actually an extra hyperparameter that may be over-optimised depending on the dataset. 
- A high AUC also looks good for the competition hosts and participants (compared to ranking problems where metrics such as precision@k are normally <0.3) that may look more daunting. Not too sure about other reasons.

Relevant links:
- http://www.chioka.in/differences-between-roc-auc-and-pr-auc/
- https://www.kaggle.com/c/ieee-fraud-detection/discussion/99982
---

4. What are other metrics that you think would also be suitable for this competition?

There are a range of other classification metrics we can choose from - below are the common ones that are useful for this problem. 

- Accuracy = TP+TN/TP+FP+FN+TN. 
    - Bad for this case because classes are imbalanced (there's so many TN we can just predict it and we'll get it >80% right); i.e if we predict that everyone is not a delinquent, our model will still get 80+% correct (on the train dataset). 
    - Therefore, our bank will lend extremely conservatively - so we might not make money + deny credit to people that may be good.
- Precision = TP/TP+FP. 
    - Good if you are an ethical person (you want a small denominator - meaning for those you predicted as positives should ideally be positives) because the costs of FPs are high here - you don't want to deny credit to someone accidentally, that's unethical. 
    - However, some defaults might happen inevitably.
- Recall = TP/TP+FN. 
    - Good if you want to minimise risk for the bank (you want a small denominator - meaning for those you predicted as positives should ideally be positives) because the costs of FNs are high here - you rather veer on the side of caution in lending to dubious people. 
    - However, you might deny credit to people that may be good.
- F1 = 2*(Recall * Precision) / (Recall + Precision). 
    - The key contrast is instead of using 1-Precision like in AUC as a variable, we use Precision. 
    - Therefore, if your precision is low, the F1 is low, and if your recall is low, your F1 score is low.

Based on this analysis, the F1 score is actually most practical for our needs - maintaining a balance between fairness and business needs. Let's walk through an example.

---

**Example:**
Let's say we want to flag relevant Delinquents out of a list of 1 million customers. Let’s say we’ve got two algorithms we want to compare with the following performance:
- Method 1: 100 predicted Delinquent, 90 actual Delinquent. 
    - Thus, TP = 90, TN = 999890, FP = 10, FN = 10.
- Method 2: 2000 predicted Delinquent, 90 actual Delinquent. 
    - Thus, TP = 90, TN = 997990, FP = 1910, FN = 10.

Clearly, Method 1’s result is preferable since they both come back with the same number of relevant results, but Method 2 brings a ton of false positives with it. The ROC measures of TPR and FPR will reflect that, but since the number of irrelevant results dwarfs the number of relevant ones, the difference is mostly lost when we calculate AUC - in this example the difference between both algorithms is only 0.0019!

- Method 1: 0.9 TPR, 0.00001 FPR
    - TPR = TP/(TP + FN) = 90/(90 + 10) = 0.9
    - FPR = FP/(FP + TN) = 10/(10 + 999890) = 0.00001
- Method 2: 0.9 TPR, 0.00191 FPR **(difference of 0.0019 in AUC scores)**
    - TPR = TP/(TP + FN) = 90/(90 + 10) = 0.9
    - FPR = FP/(FP + TN) = 1910/(1910 + 997990) = 0.0019

When using F1 score however, such differences are punished, and we want that because we don't want too many false positives! That means people are being denied credit! In this example the difference between both algorithms is ~0.85, much larger than when we compared AUC scores which fits our business case!

- Method 1: 0.9 precision, 0.9 recall. **F1 = 0.9**
    - Precision = TP/(TP + FP) = 90/(90 + 10) = 0.9
    - Recall = TP/(TP + FN) = 90/(90 + 10) = 0.9
- Method 2: 0.045 precision (difference of 0.855), 0.9 recall. **F1 = 0.0857 (difference of ~0.85)**
    - Precision = TP/(TP + FP) = 90/(90 + 1910) = 0.045
    - Recall = TP/(TP + FN) = 90/(90 + 10) = 0.9

Of course the assumption is that this is just a point in the TPR/FPR curve, but if this pattern holds (which it will because of the high FN), then clearly F1 score is a better metric for this competition.

---

5. What insight(s) do you have from your model? What is your preliminary analysis of the given dataset?

---

6. Can you get into the top 100 of the private leaderboard, or even higher?
    - **Ans:** Unfortunately no, I could only get into the top 150 with my submissions (AUC=0.86651). With more pre-processing of the data, I think it is definitely possible. I tried to hack it by fine-tuning the model, but as in any Data Science problem, the Data is the main bulk of the problem!

The detailed answers (esp. for 5) are provided in the notebook itself within the markdown cells (coloured for convenience)
