# Home_Credit_Default_Risk_System

## **Dataset Description**

***Home Credit Group is a 24 year old lending agency (founded in 1997) that provides Consumer Loans to its customers, and has operations in 9 countries in total. They entered the Indian Market in 2012 and have served more than 10 Million Consumers in the country. In order to motivate ML Engineers to construct efficient models, they have devised a Kaggle Competition for the same task. Their motto is to empower undeserved customers (by which they mean customers with little or no credit history present) by enabling them to borrow both easily as well as safely, both online as well as offline.*** 



***application_{train|test}.csv**** : 
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



