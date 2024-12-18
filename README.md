# employee_attrition_Dec24

Employee Attrition Predictor: IT Industry Synthetic Dataset
This project builds a synthetic dataset based on Gallup's Q12 Employee Engagement survey to explore employee attrition in the IT industry. It generates data reflecting employee demographics, engagement scores, workplace metrics, and attrition likelihood.

Project Overview
The purpose of this project is to:

Simulate realistic employee data for the IT industry.
Investigate the relationship between employee engagement and attrition.
Build a model that predicts employee attrition based on Gallup's Q12 survey responses and workplace metrics.
Provide a usable dataset for machine learning experiments.
The final deliverable can be deployed as an application where users can input engagement scores and receive attrition predictions.

Gallup Q12: Measuring Employee Engagement
The Gallup Q12 survey is a standardized framework that measures employee engagement using 12 key statements rated on a Likert scale (1–5):

I know what is expected of me at work.
I have the materials and equipment I need to do my work right.
At work, I have the opportunity to do what I do best every day.
In the last seven days, I have received recognition or praise for doing good work.
My supervisor, or someone at work, seems to care about me as a person.
There is someone at work who encourages my development.
At work, my opinions seem to count.
The mission or purpose of my company makes me feel my job is important.
My associates or fellow employees are committed to doing quality work.
I have a best friend at work.
In the last six months, someone at work has talked to me about my progress.
This last year, I have had opportunities at work to learn and grow.
High engagement levels (scores of 4–5) are associated with low attrition, while low engagement (scores of 1–2) increases attrition likelihood.


Dataset Structure
The generated dataset consists of 5,000 employees with the following attributes:


Column Name	and Description
Employee_ID:	Unique identifier for each employee.
Q1–Q12: 	Responses to Gallup Q12 survey questions (Likert scale: 1–5).
Age: 	Age of the employee (22–60 years).
Tenure: 	Years the employee has been with the company (derived from age).
Gender: 	Employee gender (70% Male, 30% Female).
Department: 	IT-specific departments: Development, Support, QA, DevOps, Management.
Salary:	Annual salary (ranging from 50,000 to 150,000).
Overtime:	Whether the employee works overtime (Yes/No, 40% chance for 'Yes').
Distance_to_Work:	Commuting distance in kilometers (normally distributed around 15 km).
Attrition:	Whether the employee left the company (Yes/No).


