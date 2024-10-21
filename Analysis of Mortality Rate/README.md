# Do Right Handers Live Longer Than Left Handers

# About :
Handedness are the preference for using one hand over the other, is a fascinating aspect of human biology. This difference in handedness has been the subject of much research, with scientists exploring the underlying neurological and genetic factors that contribute to it. One area of particular interest has been the potential link between handedness and mortality.
 
In this project we will explore this phenomenon using age distribution data to see if we can reproduce a difference in average age at death purely from the changing rates of left-handedness over time, refuting the claim of early death for left-handers. This project uses pandas and Bayesian statistics to analyze the probability of being a certain age at death given that you are reported as left-handed or right-handed.

A National Geographic survey in 1986 resulted in over a million responses that included age, sex, and hand preference for throwing and writing. Researchers Avery Gilbert and Charles Wysocki analyzed this data and noticed that rates of left-handedness were around 13% for people younger than 40 but decreased with age to about 5% by the age of 80. They concluded based on analysis of a subgroup of people who throw left-handed but write right-handed that this age-dependence was primarily due to changing social acceptability of left-handedness. This means that the rates aren't a factor of age specifically but rather of the year you were born, and if the same study was done today, we should expect a shifted version of the same distribution as a function of age. Ultimately, we'll see what effect this changing rate has on the apparent mean age of death of left-handed people, but let's start by plotting the rates of left-handedness as a function of age.

----

# Platforms Used :
## JUPYTER:
Jupyter, formerly known as IPython, is an open-source web-based interactive computing environment that offers a user-friendly interface for executing code, creating visualizations, and writing narrative text. It has gained widespread popularity among data scientists, researchers, and educators due to its versatility and ease of use.

Jupyter's popularity in data science stems from its ability to seamlessly integrate code, text, and visualizations within a single document, known as a Jupyter notebook. This unique feature enables data scientists to effectively communicate their findings and insights, making it an invaluable tool for collaboration and knowledge sharing.

## Interactive Coding and Exploration:
Jupyter allows for interactive coding, enabling data scientists to execute code cell by cell and observe the results immediately. This iterative approach facilitates rapid prototyping and experimentation, allowing data scientists to refine their code and algorithms efficiently.

## Rich Data Visualization:
Jupyter seamlessly integrates with various data visualization libraries, such as Matplotlib, Seaborn, and Plotly. Data scientists can create interactive plots, charts, and graphs directly within the notebook, enhancing the understanding and communication of complex data patterns and trends.

---

# IMPLEMENTATION :
In this project, you will explore this phenomenon using age distribution data to see if we canreproduce a difference in average age at death purely from the changing rates of left-handednessover time, refuting the claim of early death for left-handers. This notebook uses pandas and Bayesianstatistics to analyze the probability of being a certain age at death given that you are reported as left-handed or right-handed

"Gathering Requirements and Defining Problem Statement":

To investigate whether there is a significant difference in the average lifespan of left-handed individuals compared to right-handed individuals.

---
