# Problem Statement:-




An education company named `X Education` sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website and browse for courses.

🎓 **X Education Lead Conversion Process** 📈


1. **Traffic Generation**: 🌐 The company markets its courses on various platforms such as websites, search engines, and social media to attract industry professionals interested in online courses.

2. **Website Engagement**: 🖥️ Prospective customers land on the X Education website and explore available courses, watch videos, and gather information about the offerings.

3. **Lead Acquisition**: 📝 Interested individuals fill out forms with their contact details (email address or phone number) to express interest in specific courses. Leads can also come through past referrals.

4. **Lead Identification**: 🔍 The company identifies potential leads from the pool of acquired leads based on certain criteria like engagement level, course interest, and past interactions.

5. **Sales Communication**: 📞 The sales team initiates communication with potential leads through phone calls, emails, and other channels to nurture the relationship and guide them towards making a purchase.

6. **Conversion to Sales**: 💼 Some leads are successfully converted into sales by the sales team's efforts, resulting in course purchases and revenue generation for X Education.

By focusing efforts on identifying and prioritizing potential leads, X Education aims to improve its lead-to-sale conversion rate and enhance overall sales performance.


Now, although `X Education` gets a lot of leads, its lead-to-sale conversion rate is very poor. For example, if they acquire 100 leads in a day, only about 30 of them are converted into successful sales. To make this process more efficient, the company wishes to identify the most potential leads, also known as ‘Hot Leads’. If they successfully identify this set of leads, the lead conversion rate would go up as the sales team would now be focusing more on communicating with the potential leads rather than making calls to everyone. A typical lead conversion process can be represented using the following funnel.

<p align='center' ><img src="https://cdn.upgrad.com/UpGrad/temp/189f213d-fade-4fe4-b506-865f1840a25a/XNote_201901081613670.jpg"></p>

As you can see, there are a lot of leads generated in the initial stage (the initial pool of leads), but only a few of them come out as paying customers from the bottom (converted leads). In the middle stage (lead nurturing), you need to nurture the potential leads well (i.e., educate the leads about the product, constantly communicate, etc.) in order to get a higher lead conversion.

 

`X Education` has appointed you to help them select the most promising leads, i.e., the leads that are most likely to convert into paying customers. The company requires you to build a model wherein you need to assign a lead score to each of the leads such that the customers with a higher lead score have a higher conversion chance and the customers with a lower lead score have a lower conversion chance. The CEO, in particular, has given a ballpark estimate of the target lead conversion rate as being around 80%.

The target variable, in this case, is the column `Converted`, which tells whether a past lead was converted or not, where 1 means it was converted and 0 means it wasn’t converted. 


## Goals of the Case Study
There are quite a few goals for this case study. They are as follows:-

- Build a logistic regression model to assign a lead score between 0 and 100 to each of the leads, which can be used by the company to target potential leads. A higher score would mean that the lead is hot, i.e., most likely to convert, whereas a lower score would mean that the lead is cold and will mostly not get converted.

- There are some more problems presented by the company that your model should be able to adjust to if the company’s requirements change in the future, so you will need to handle these as well. 

- These problems are provided in a separate doc file. Please fill it out based on the logistic regression model you got in the first step. Also, make sure you include this in your final PowerPoint presentation, where you’ll make recommendations.