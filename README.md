# Home_Credit_Default_Risk_System

## **Dataset Description**

***Home Credit Group is a 24 year old lending agency (founded in 1997) that provides Consumer Loans to its customers, and has operations in 9 countries in total. They entered the Indian Market in 2012 and have served more than 10 Million Consumers in the country. In order to motivate ML Engineers to construct efficient models, they have devised a Kaggle Competition for the same task. Their motto is to empower undeserved customers (by which they mean customers with little or no credit history present) by enabling them to borrow both easily as well as safely, both online as well as offline.*** 



***application_{train|test}.csv*** : 
* This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
* Static data for all applications. One row represents one loan in our data sample.

***bureau.csv*** : 
* All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
* For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.

***bureau_balance.csv*** : 
* Monthly balances of previous credits in Credit Bureau.
* This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.

***POS_CASH_balance.csv*** :
* Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
* This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.

***credit_card_balance.csv*** :
* Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
* This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.

***previous_application.csv*** :
* All previous applications for Home Credit loans of clients who have loans in our sample.
* There is one row for each previous application related to loans in our data sample.

***installments_payments.csv*** :
* Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
* There is a) one row for every payment that was made plus b) one row each for missed payment.
* One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

***HomeCredit_columns_description.csv*** :
* This file contains descriptions for the columns in the various data files.
Data

![image](https://user-images.githubusercontent.com/101316217/215870604-2b58c751-dafa-4cb0-a7fe-4ae25365a155.png)


**Constraints:**

**(i) Interpretability is Important:**

This means that we should be able to generate the Probability Estimates, of an applicant being capable or not capable of repayment of a loan, rather than strictly classifying the applicant as either of them. However, interpretability of the model is not as important as in the case of Medical Applications like Cancer Diagnosis.
Eg: If probability is 0.5 for an applicant’s capability and 0.9 in the other case, we can very well conclude that we are much more sure of the capability when the value is 0.9 (and classified as 1) rather than when the value is 0.5 (and then classified as 1).

**(ii) No Strict Latency Constraints:**

This is not exactly a Low-Latency Requirement because in a Low Latency Requirement Problem, such as for Internet Companies, low latency refers to a few Milliseconds.
Our algorithm over here can take sometime to run in order to ensure high accuracy in predicting repayment capabilities. The Bank/Financial Institution doesn’t need to deliver the results in a very quick time.

**(iii) High Misclassification Cost:**

This is a very important real world metric that needs to be considered because our cost of misclassification can be very high.
If a loan applicant who is not capable of loan repayment is classified as capable and he/she is granted a loan, and in case he/she is unable to repay the loan, the bank or financial institution runs into delinquencies and may suffer losses, which could even have to be Written Off.
Similarly, if a capable applicant is classified as non-capable, the person has his/her application rejected and the Bank loses out on a customer, which affects their profits.

***Descrription of columns is included in HomeCredit_columns_description.csv*** 

## EDA

Note that the output of the EDA notebook was removed because the final size of the notebook was very high
![newplot (1)](https://user-images.githubusercontent.com/101316217/216818171-5bf72086-7352-4785-8163-6b1afe55b46a.png)
![newplot (2)](https://user-images.githubusercontent.com/101316217/216818176-4a594bbe-4ff3-451f-b06a-69506f0f184a.png)
![newplot (3)](https://user-images.githubusercontent.com/101316217/216818178-aba68e56-ea38-4cba-9271-fd7a1ff369ee.png)
![newplot (4)](https://user-images.githubusercontent.com/101316217/216818183-ef1b3b9d-1273-430d-8334-3a8f2ffcc2dd.png)
![newplot (5)](https://user-images.githubusercontent.com/101316217/216818187-3f91790f-73af-4c97-85d2-32e322c88db9.png)
![newplot (6)](https://user-images.githubusercontent.com/101316217/216818194-bf6561f9-157e-498a-9c64-93d2578d7074.png)
![newplot (7)](https://user-images.githubusercontent.com/101316217/216818199-7f3ec424-3c6c-4090-bcac-35c76405544b.png)
![newplot (8)](https://user-images.githubusercontent.com/101316217/216818201-92743db7-c1d3-499c-98db-00b2525bfcd2.png)
![newplot (9)](https://user-images.githubusercontent.com/101316217/216818204-4a51a5d2-9fe4-4ba2-9e1a-50e35001f82a.png)
![newplot (10)](https://user-images.githubusercontent.com/101316217/216818207-bf33f522-4365-44d5-8405-c701e8e23426.png)
![newplot (11)](https://user-images.githubusercontent.com/101316217/216818209-6a65a06b-193e-4170-b01c-7e5774c2ba08.png)
![newplot (12)](https://user-images.githubusercontent.com/101316217/216818214-2b158d36-e963-4ff9-b346-79a752c956b0.png)
![newplot (13)](https://user-images.githubusercontent.com/101316217/216818217-48861c23-5e61-4a61-91af-06ff20b98653.png)
![newplot (14)](https://user-images.githubusercontent.com/101316217/216818220-3ac65e30-3f39-413e-bee3-364380e4efc5.png)
![newplot (15)](https://user-images.githubusercontent.com/101316217/216818226-cf216273-152a-4c7f-8bc1-693b7f2763c3.png)
![newplot (16)](https://user-images.githubusercontent.com/101316217/216818229-e5b0593b-ec91-4aec-a392-354534765a59.png)
![newplot (17)](https://user-images.githubusercontent.com/101316217/216818234-7938d846-e28b-4e24-bb70-449728295bba.png)
![newplot (18)](https://user-images.githubusercontent.com/101316217/216818235-f4c07711-e3a1-4dce-be21-4eb8729b683c.png)
![newplot (19)](https://user-images.githubusercontent.com/101316217/216818241-03e0fab6-947a-491f-a389-1b7bd9815040.png)
![newplot (20)](https://user-images.githubusercontent.com/101316217/216818243-1438d9f1-ae47-4e39-83ef-c9997aec728d.png)
![newplot (21)](https://user-images.githubusercontent.com/101316217/216818248-73dd71a5-3c78-4e0d-8d0c-3da9ef8ed310.png)
![newplot (22)](https://user-images.githubusercontent.com/101316217/216818251-34e8d129-7f91-43c7-bd9f-81d241ba37ae.png)


## Output 
![image](https://user-images.githubusercontent.com/101316217/216772940-c96b1b99-8def-4307-bbd4-e7e537be50c4.png)

## Future Work
Try Other models such as Random Forest, or Xgboost

Try to get more data using data aggregation from other files included

Try to investigate more in other files and get more insights about the data

Tunning the model

Try another data balancing techniques such as SMOTE.

Try different combination of scaling techniques such as standard scaler or MinMax scaler.


## References 

[1] M. Qamruzzaman and W. Jianguo, “Financial innovation and economic growth in Bangladesh,” Financ. Innov., 2017, doi: 10.1186/s408540-0070-017-.

[2] F. D. I. Corporation, “2017 FDIC national survey of unbanked and underbanked households.” Federal Deposit Insurance Corporation Washington, DC, 2018.

[3] M. Alam, “Risk prediction of loan default using knowledge graph,” 2022.

[4] J. P. Simon, “Artificial intelligence: scope, players, markets and geography,” Digit. Policy, Regul. Gov. ,2019, doi: 10.1108/DPRG-080039-2018-.

[5] C. Janiesch, P. Zschech, and K. Heinrich, “Machine learning and deep learning,” Electron. Mark., 2021, doi: 10.1007/s125252-00475-021-.

[6] U. Kose, “Using artificial intelligence techniques for economic time series prediction,” in Contemporary Studies in Economic and Financial Analysis, 2019. doi: 10.1108/S1569375920190000101002-.

[7] K. Mungai and A. Bayat, “The impact of big data on the South African banking industry,” 2018.

[8] V. Dhar, “Data science and prediction,” Commun. ACM, 2013, doi: 10.11452500499/.

[9] A. Nabil, “Data Science in FinTech: credit risk prediction using Deep Learning,” ETSI_Informatica, 2020.

[10] Home Credit Group, “Home Credit Default Risk DataSet,” Kaggle, 2018.

[11] P. Beck, “Predicting Loan Default Likelihood Using Machine Learning,” 2021.

[12] Y. E. Gundogmus, M. Nuhuz, and M. Tez, “Riskbased Fraud Analysis for Bank Loans with Autonomous Machine Learning,” in y-BIS 2019 Conference Book: Recent Advances n Data Sc ence and Bus ness Analyt cs, 2019, p. 143.

[13] X. Chen, X. Liu, Z. Liu, P. Song, and M. Zhong, “A deep learning approach using DeepGBM for credit assessment,” 2019. doi: 10.11453366194.3366333/.

[14] Dall›asta Rigo, Elif Yağmur. Evaluation of stacking for predicting credit risk scores. Diss. Applied Data Science, 2020.

[15] Y. Tounsi, H. Anoun, and L. Hassouni, “CSMAS: Improving Multi-Agent Credit Scoring System by Integrating Big Data and the new generation of Gradient Boosting Algorithms,” 2020. doi: 10.11453386723.3387851/.

[16] G. Ke et al., “LightGBM: A highly efficient gradient boosting decision tree,” 2017.

[17] C. Egan, “Improving Credit Default Prediction Using Explainable AI,” Dublin, National College of Ireland, 2021.

[18] Rodríguez P, Bautista MA, Gonzalez J, Escalera S. Beyond one-hot encoding: Lower dimensional target embedding. Image and Vision Computing. 2018 Jul 1;75:2131-.

[19] Boubiche S, Boubiche DE, Bilami A, Toral-Cruz H. Big data challenges and data aggregation strategies in wireless sensor networks. IEEE access. 2018 May 3;6:2055871-.

[20] Fushiki T. Estimation of prediction error by using K-fold cross-validation. Statistics and Computing. 2011 Apr;21(2):13746-.

https://www.kaggle.com/competitions/home-credit-default-risk/code

https://www.kaggle.com/competitions/home-credit-default-risk/overview

https://www.kaggle.com/competitions/home-credit-default-risk/data

https://www.kaggle.com/rinnqd/reduce-memory-usage

https://academic.oup.com/bioinformatics/article/26/10/1340/193348

https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
